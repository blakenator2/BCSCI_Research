import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.draw    import polygon
from scipy           import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster         import KMeans
from sklearn.preprocessing   import StandardScaler
from sklearn.utils           import class_weight
from sklearn.ensemble        import RandomForestClassifier, IsolationForest
from sklearn.metrics         import classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from numba import jit
from multiprocessing import Pool, cpu_count
from functools import partial

import tensorflow as tf
from tensorflow.keras        import layers, models, Input, callbacks
import matplotlib.patches    as mpatches
import matplotlib.colors     as mcolors

# === PARAMETERS (constants are safe) ===
WINDOW, HALF = 25, 24
TRAIN_DAYS = list(range(1203, 1303))
TEST_DAYS  = list(range(1305, 1405))
ALL_DAYS   = TRAIN_DAYS + TEST_DAYS
STEP       = 3
RNG        = 42
KM         = 10
EPOCHS     = 40
BATCH      = 128
RND_WEIGHT = 2.0
REGIONS9   = ["NW","N","NE","W","C","E","SW","S","SE"]
ROI_TYPES  = ["COL","CL","COH","NROI"]
TYPE_COLORS = {"COL":"red","CL":"blue","COH":"green","NROI":"orange"}

# === FUNCTIONS (must be defined at module level for pickling) ===

def create_roi_mask(df, day, shape):
    mask = np.zeros(shape, dtype=np.int8)
    day_data = df[df["Day"]==day]
    for name, grp in day_data.groupby("Name"):
        pts = grp[["x","y"]].values
        if len(pts)<3:
            continue
        xs, ys = pts[:,0].astype(np.int32), pts[:,1].astype(np.int32)
        rows = (shape[0]-1) - ys
        cols = xs
        rr, cc = polygon(rows, cols, shape=shape)
        code = ROI_TYPES.index(grp["Label"].iat[0]) + 1
        mask[rr,cc] = code
    return mask

def load_grids(day):
    P = pd.read_csv(f"pressure/day{day}.txt", sep=r"\s+", header=None).values.astype(np.float32)
    W = pd.read_csv(f"wind/day{day}.txt",    sep=r"\s+", header=None).values.astype(np.float32)
    return P, W

@jit(nopython=True, cache=True)
def extract_window_safe(arr, x, y_idx, corner_idx, H, W):
    """Extract 25x25 window with boundary handling"""
    if corner_idx == 0:  # NW
        r0, r1, c0, c1 = y_idx-HALF, y_idx+1, x-HALF, x+1
    elif corner_idx == 1:  # NE
        r0, r1, c0, c1 = y_idx-HALF, y_idx+1, x, x+HALF+1
    elif corner_idx == 2:  # SW
        r0, r1, c0, c1 = y_idx, y_idx+HALF+1, x-HALF, x+1
    else:  # SE
        r0, r1, c0, c1 = y_idx, y_idx+HALF+1, x, x+HALF+1
    
    rr0, rr1 = max(0, r0), min(H, r1)
    cc0, cc1 = max(0, c0), min(W, c1)
    
    result = np.zeros((WINDOW, WINDOW), dtype=np.float32)
    
    result_r0 = max(0, -r0)
    result_c0 = max(0, -c0)
    result_r1 = result_r0 + (rr1 - rr0)
    result_c1 = result_c0 + (cc1 - cc0)
    
    for i in range(rr1 - rr0):
        for j in range(cc1 - cc0):
            result[result_r0 + i, result_c0 + j] = arr[rr0 + i, cc0 + j]
    
    if result_r0 > 0:
        for i in range(result_r0):
            for j in range(WINDOW):
                result[i, j] = result[result_r0, j]
    if result_r1 < WINDOW:
        for i in range(result_r1, WINDOW):
            for j in range(WINDOW):
                result[i, j] = result[result_r1-1, j]
    if result_c0 > 0:
        for i in range(WINDOW):
            for j in range(result_c0):
                result[i, j] = result[i, result_c0]
    if result_c1 < WINDOW:
        for i in range(WINDOW):
            for j in range(result_c1, WINDOW):
                result[i, j] = result[i, result_c1-1]
    
    return result

@jit(nopython=True, cache=True)
def compute_basic_features(pwin, wwin, psx, psy, wsx, wsy):
    """Compute basic statistical features"""
    pmean = np.mean(pwin)
    pstd = np.std(pwin)
    pcenter = pwin[HALF, HALF]
    
    wmean = np.mean(wwin)
    wstd = np.std(wwin)
    wcenter = wwin[HALF, HALF]
    
    return np.array([pmean, pstd, psx, psy, pcenter, wmean, wstd, wsx, wsy, wcenter], dtype=np.float32)

def tex_fast(mat):
    """Optimized texture computation"""
    mat_range = mat.max() - mat.min()
    if mat_range < 1e-6:
        return [0.0, 0.0, 1.0, 1.0]
    
    a = ((mat - mat.min()) / mat_range * 255).astype(np.uint8)
    G = graycomatrix(a, [1], [0], levels=256, symmetric=True, normed=True)
    
    return [
        graycoprops(G, 'contrast')[0,0],
        graycoprops(G, 'dissimilarity')[0,0],
        graycoprops(G, 'homogeneity')[0,0],
        graycoprops(G, 'energy')[0,0]
    ]

@jit(nopython=True, cache=True)
def get_region_code(x, y):
    """Return region index 0-8"""
    if y >= 50:
        if x < 25: return 0  # NW
        elif x < 50: return 1  # N
        else: return 2  # NE
    elif y >= 25:
        if x < 25: return 3  # W
        elif x < 50: return 4  # C
        else: return 5  # E
    else:
        if x < 25: return 6  # SW
        elif x < 50: return 7  # S
        else: return 8  # SE

def process_day(day, roi_df_subset):
    """Process a single day - designed to run in parallel"""
    prev = day - 1
    
    try:
        P, W = load_grids(prev)
    except FileNotFoundError:
        return None
    
    H, Wd = P.shape
    mask = create_roi_mask(roi_df_subset, day, (H, Wd))
    
    # Pre-compute Sobel gradients
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 8.0
    
    P_sobel_x = ndi.convolve(P, sobel_x, mode='constant')
    P_sobel_y = ndi.convolve(P, sobel_y, mode='constant')
    W_sobel_x = ndi.convolve(W, sobel_x, mode='constant')
    W_sobel_y = ndi.convolve(W, sobel_y, mode='constant')
    
    # Collect data for this day
    day_img_patches = []
    day_feat_list = []
    day_lbl_list = []
    day_coords_list = []
    day_types_list = []
    day_regions_list = []
    
    xr_coords = np.arange(0, Wd, STEP)
    yr_coords = np.arange(0, H, STEP)
    
    for xr in xr_coords:
        for yr in yr_coords:
            row = H - 1 - yr
            code = mask[row, xr]
            is_roi = code > 0
            lbl = 0 if is_roi else 1
            rtype = ROI_TYPES[code-1] if is_roi else "random"
            
            psx, psy = P_sobel_x[row, xr], P_sobel_y[row, xr]
            wsx, wsy = W_sobel_x[row, xr], W_sobel_y[row, xr]
            
            for corner_idx in range(4):
                pwin = extract_window_safe(P, xr, row, corner_idx, H, Wd)
                wwin = extract_window_safe(W, xr, row, corner_idx, H, Wd)
                
                day_img_patches.append(np.stack([pwin, wwin], -1))
                
                basic = compute_basic_features(pwin, wwin, psx, psy, wsx, wsy)
                tex_p = tex_fast(pwin)
                tex_w = tex_fast(wwin)
                day_feat_list.append(np.concatenate([basic, tex_p, tex_w]))
                
                day_lbl_list.append(lbl)
                day_coords_list.append((xr, yr))
                day_types_list.append(rtype)
                day_regions_list.append(get_region_code(xr, yr))
    
    return {
        'day': day,
        'img_patches': np.array(day_img_patches, dtype=np.float32),
        'features': np.array(day_feat_list, dtype=np.float32),
        'labels': np.array(day_lbl_list, dtype=np.int32),
        'coords': np.array(day_coords_list, dtype=np.int32),
        'types': np.array(day_types_list),
        'regions': np.array(day_regions_list, dtype=np.int32)
    }

# === MAIN EXECUTION ===
if __name__ == '__main__':
    # Calculate ms and NUM_WORKERS inside the main block
    ms = int(round(time.time() * 1000))
    NUM_WORKERS = max(1, cpu_count() - 1)
    
    print(f"Using {NUM_WORKERS} worker processes")
    
    # Load ROI data
    roi_df = pd.read_csv("roi_data_with_status.csv")
    required = {"Day","Name","x","y","Label"}
    if not required.issubset(roi_df.columns):
        raise RuntimeError(f"roi_data_with_status.csv must contain columns: {required}")
    
    # === PARALLEL COLLECTION ===
    print("Collecting patches and features in parallel...")
    start_time = time.time()
    
    # Filter days
    valid_days = []
    for day in ALL_DAYS:
        prev = day - 1
        if prev >= min(ALL_DAYS):
            valid_days.append(day)
    
    # Create partial function with roi_df baked in
    process_day_with_roi = partial(process_day, roi_df_subset=roi_df)
    
    # Process in parallel
    with Pool(processes=NUM_WORKERS) as pool:
        results = []
        for i, result in enumerate(pool.imap(process_day_with_roi, valid_days)):
            if result is not None:
                results.append(result)
            if (i + 1) % 5 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {i + 1}/{len(valid_days)} days in {elapsed:.1f}s ({elapsed/(i+1):.2f}s per day)")
    
    print(f"Parallel processing complete in {time.time()-start_time:.1f}s")
    
    # === MERGE RESULTS ===
    print("Merging results...")
    img_patches = np.vstack([r['img_patches'] for r in results])
    X_basic = np.vstack([r['features'] for r in results])
    y_all = np.concatenate([r['labels'] for r in results])
    coords = np.vstack([r['coords'] for r in results])
    types_arr = np.concatenate([r['types'] for r in results])
    regions_codes = np.concatenate([r['regions'] for r in results])
    
    # Reconstruct day list
    days_arr = np.concatenate([np.full(len(r['labels']), r['day'], dtype=np.int32) for r in results])
    
    print(f"Collected {len(img_patches)} patches total")
    
    # === CONVERT TO FINAL FORMAT ===
    X_img = img_patches
    
    # Convert region codes to one-hot
    reg_ohe = np.zeros((len(regions_codes), 9), dtype=np.float32)
    reg_ohe[np.arange(len(regions_codes)), regions_codes] = 1
    
    # Rebuild regions array for compatibility
    regions = np.array([REGIONS9[c] for c in regions_codes])
    
    # === K-MEANS ===
    print("Running K-means clustering...")
    km = KMeans(n_clusters=KM, random_state=RNG, n_init=10).fit(X_basic)
    clus_ohe = np.zeros((len(X_basic), KM), dtype=np.float32)
    clus_ohe[np.arange(len(X_basic)), km.labels_] = 1
    dists = np.linalg.norm(X_basic - km.cluster_centers_[km.labels_], axis=1).reshape(-1, 1)
    
    X_feat = np.hstack([X_basic, reg_ohe, clus_ohe, dists])
    X_feat = StandardScaler().fit_transform(X_feat).astype(np.float32)
    
    # === SPLIT BY DAY ===
    train_mask = np.isin(days_arr, TRAIN_DAYS)
    test_mask  = np.isin(days_arr, TEST_DAYS)
    
    X_img_tr,  X_img_te   = X_img [train_mask], X_img [test_mask]
    X_feat_tr, X_feat_te  = X_feat[train_mask], X_feat[test_mask]
    y_tr,      y_te       = y_all[train_mask],  y_all[test_mask]
    coords_tr, coords_te  = coords [train_mask], coords [test_mask]
    types_tr,  types_te   = types_arr[train_mask], types_arr[test_mask]
    days_tr,   days_te    = days_arr[train_mask], days_arr[test_mask]
    
    print("Train class counts:", np.bincount(y_tr))
    print("Test  class counts:", np.bincount(y_te))
    
    # === ROBUST CLASS WEIGHTS ===
    classes_present = np.unique(y_tr)
    print("Classes present in y_tr:", classes_present)
    
    if classes_present.size == 0:
        raise RuntimeError("No training labels found. Check day ranges and data files.")
    elif classes_present.size == 1:
        print("WARNING: Only one class present in training set. Using neutral weight 1.0.")
        cw = {int(classes_present[0]): 1.0}
    else:
        cw_arr = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=classes_present,
            y=y_tr
        )
    
    print(f"\nTotal time: {time.time()-start_time:.1f}s")

    # Optionally boost RANDOM (label 1) if present
    if 1 in classes_present:
        idx = np.where(classes_present == 1)[0][0]
        cw_arr[idx] *= RND_WEIGHT
    cw = {int(c): float(w) for c, w in zip(classes_present, cw_arr)}

    # === BUILD & TRAIN DNN ===
    counts = np.bincount(y_tr, minlength=2)
    neg, pos = counts[0], counts[1]
    bias_init = tf.keras.initializers.Constant(
    np.log((pos + 1e-6) / (neg + 1e-6)).astype(np.float32)
    )

    # Define a Sequential DNN
    dnn = models.Sequential([
        Input(shape=(X_feat_tr.shape[1],)),      # input size = num features
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid", bias_initializer=bias_init)  # binary output
    ])

    # Compile model
    dnn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Train model
    history = dnn.fit(
        X_feat_tr, y_tr,
        validation_data=(X_feat_te, y_te),
        epochs=EPOCHS,
        batch_size=BATCH,
        class_weight=cw,   # you computed cw earlier
        verbose=1
    )

    # Evaluate on test set
    loss, acc = dnn.evaluate(X_feat_te, y_te, verbose=0)
    print(f"Test accuracy: {acc:.3f}")

    # --- Train Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, class_weight=cw, random_state=RNG)
    rf.fit(X_feat_tr, y_tr)

    # --- Predict probabilities ---
    # DNN outputs shape (n_samples, 1), flatten to (n_samples,)
    p_dnn = dnn.predict(X_feat_te).ravel()

    # RF probability for class 0 (ROI) or 1? 
    # sklearn's predict_proba: [:,1] is usually "positive class" (label=1)
    # Since in your labeling 0 = ROI, 1 = random, we want probability of ROI (0)
    # So we can use column 0
    p_rf = rf.predict_proba(X_feat_te)[:, 0]  

    # Isolation Forest contribution (all zeros)
    p_if = np.zeros_like(p_rf)

    # --- Ensemble (weighted) ---
    p_ens = 0.4 * p_dnn + 0.4 * p_rf + 0.2 * p_if

    # --- Find best threshold for ROI detection ---
    best_acc, best_thr = 0, 0.5
    for t in np.linspace(0.1, 0.9, 81):
        y_pred = (p_ens >= t).astype(int)  # binary predictions: 0=ROI, 1=random

        # Compute confusion matrix
        cm = confusion_matrix(y_te, y_pred, labels=[0,1])
        
        # Recall per class
        rec_roi    = cm[0,0] / cm[0].sum() if cm[0].sum() else 0.0
        rec_random = cm[1,1] / cm[1].sum() if cm[1].sum() else 0.0

        # Overall accuracy
        acc = (y_pred == y_te).mean()

        # Update best threshold
        if rec_roi >= 0.5 and rec_random >= 0.5 and acc > best_acc:
            best_acc, best_thr = acc, t

    print(f"Best threshold for ROI detection: {best_thr:.2f}, accuracy: {best_acc:.3f}")

    # === TYPE-ONLY CONFUSION MATRIX: COL / COH / CL ===
    print("\n=== Training Multi-Class Type Classifier ===")

    # Filter to only ROI samples (label == 0)
    roi_train_mask = (y_tr == 0)
    roi_test_mask = (y_te == 0)

    X_feat_roi_tr = X_feat_tr[roi_train_mask]
    X_feat_roi_te = X_feat_te[roi_test_mask]
    types_roi_tr = types_tr[roi_train_mask]
    types_roi_te = types_te[roi_test_mask]

    # Convert type strings to numeric labels
    type_to_label = {"COL": 0, "COH": 1, "CL": 2, "NROI": 3}
    label_to_type = {v: k for k, v in type_to_label.items()}

    y_type_tr = np.array([type_to_label[t] for t in types_roi_tr])
    y_type_te = np.array([type_to_label[t] for t in types_roi_te])

    print(f"ROI training samples: {len(y_type_tr)}")
    print(f"ROI test samples: {len(y_type_te)}")
    print("Type distribution (train):", {label_to_type[i]: (y_type_tr==i).sum() for i in range(4)})

    # Train Random Forest for type classification
    type_cw = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_type_tr),
        y=y_type_tr
    )
    type_cw_dict = {int(c): float(w) for c, w in zip(np.unique(y_type_tr), type_cw)}

    rf_type = RandomForestClassifier(
        n_estimators=200, 
        class_weight=type_cw_dict, 
        random_state=RNG,
        max_depth=20
    )
    rf_type.fit(X_feat_roi_tr, y_type_tr)

    # Predict types on test ROI samples
    y_type_pred = rf_type.predict(X_feat_roi_te)
    pred_types_strings = [label_to_type[int(p)] for p in y_type_pred]
    true_types_strings = [label_to_type[int(t)] for t in y_type_te]

    # === NOW CREATE PROPER CONFUSION MATRICES ===

    # 3x3 matrix (COL/COH/CL only)
    labels3 = ["COL", "COH", "CL"]
    mask_3class = np.array([t in labels3 for t in true_types_strings])
    cm_types_3 = confusion_matrix(
        np.array(true_types_strings)[mask_3class],
        np.array(pred_types_strings)[mask_3class],
        labels=labels3
    )

    plt.figure(figsize=(5, 4))
    plt.imshow(cm_types_3, cmap="Blues")
    plt.title("ROI Type Confusion (COL/COH/CL)")
    plt.xticks(range(len(labels3)), labels3)
    plt.yticks(range(len(labels3)), labels3)
    plt.xlabel("Predicted Type")
    plt.ylabel("True Type")
    for i in range(len(labels3)):
        for j in range(len(labels3)):
            val = cm_types_3[i, j]
            plt.text(j, i, val, ha="center", va="center",
                    color="white" if val > cm_types_3.max()/2 else "black")
    plt.tight_layout()
    plt.show()

    # 4x4 matrix (all ROI types including NROI)
    labels4 = ["COL", "COH", "CL", "NROI"]
    cm_types_4 = confusion_matrix(true_types_strings, pred_types_strings, labels=labels4)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm_types_4, cmap="Blues")
    plt.title("ROI Type Confusion (All Types)")
    plt.xticks(range(len(labels4)), labels4)
    plt.yticks(range(len(labels4)), labels4)
    plt.xlabel("Predicted Type")
    plt.ylabel("True Type")
    for i in range(len(labels4)):
        for j in range(len(labels4)):
            val = cm_types_4[i, j]
            plt.text(j, i, val, ha="center", va="center",
                    color="white" if val > cm_types_4.max()/2 else "black")
    plt.tight_layout()
    plt.show()

    # Per-type metrics
    from sklearn.metrics import classification_report
    print("\n=== ROI Type Classification Report ===")
    print(classification_report(true_types_strings, pred_types_strings, labels=labels4))

    # Calculate per-type recall
    support = np.array([(np.array(true_types_strings) == t).sum() for t in labels4], dtype=float)
    diag = np.diag(cm_types_4).astype(float)
    recall_by_type = {t: (d / s if s > 0 else 0.0) for t, d, s in zip(labels4, diag, support)}
    print("\nPer-type recall:", recall_by_type)

    # F1 score
    from sklearn.metrics import f1_score
    print(f"Weighted F1 score: {f1_score(true_types_strings, pred_types_strings, average='weighted', labels=labels4):.3f}")

    # === TRUE vs. PREDICTED ROI MASKS (color‐coded by type) ===
    cmap_types = mcolors.ListedColormap(["lightgrey"] + [TYPE_COLORS[t] for t in ROI_TYPES])
    type_code  = {t:i+1 for i,t in enumerate(ROI_TYPES)}

    for day in TEST_DAYS:
        P,_        = load_grids(day-1)
        H,W        = P.shape
        true_mask  = create_roi_mask(roi_df,day,(H,W))
        sel        = (days_te==day)
        coords_day = coords_te[sel]
        types_day  = types_te[sel]
        preds_day  = y_pred[sel]

        # build predicted‐type mask: wherever we predicted ROI (label 0)
        pred_mask = np.zeros((H,W),dtype=int)
        for (x,y),tp,pr in zip(coords_day,types_day,preds_day):
            if pr==0 and tp in type_code:            # only colour real ROI types
                pred_mask[H-1-y, x] = type_code[tp]

        fig,(axT,axP) = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
        axT.imshow(true_mask, cmap=cmap_types, origin="upper", vmin=0, vmax=len(ROI_TYPES))
        axT.set_title(f"True mask — Day {day}");  axT.axis("off")
        axP.imshow(pred_mask, cmap=cmap_types, origin="upper", vmin=0, vmax=len(ROI_TYPES))
        axP.set_title(f"Predicted mask — Day {day}"); axP.axis("off")

        patches = [mpatches.Patch(color=cmap_types(i), label=l)
                for i,l in enumerate(["BG"]+ROI_TYPES)]
        axP.legend(handles=patches, bbox_to_anchor=(1.05,1), loc="upper left", title="ROI Type")
        plt.tight_layout()
        plt.show()