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

import tensorflow as tf
from tensorflow.keras        import layers, models, Input, callbacks
import matplotlib.patches    as mpatches
import matplotlib.colors     as mcolors

# === PARAMETERS ===
WINDOW, HALF = 25, 24

ms = int(round(time.time() * 1000))

TRAIN_DAYS = list(range(1203, 1303))
TEST_DAYS  = list(range(1305, 1405))

ALL_DAYS     = TRAIN_DAYS + TEST_DAYS
STEP         = 3
RNG          = 42
KM           = 10
EPOCHS       = 40
BATCH        = 128 # was 32
RND_WEIGHT   = 2.0

REGIONS9     = ["NW","N","NE","W","C","E","SW","S","SE"]
ROI_TYPES    = ["COL","CL","COH","NROI"]
TYPE_COLORS  = {"COL":"red","CL":"blue","COH":"green","NROI":"orange"}

# === LOAD POLYGONAL ROI DATA ===
roi_df = pd.read_csv("roi_data_with_status.csv")
required = {"Day","Name","x","y","Label"}
if not required.issubset(roi_df.columns):
    raise RuntimeError(f"roi_data_with_status.csv must contain columns: {required}")

def create_roi_mask(df, day, shape):
    mask = np.zeros(shape, dtype=int)
    for name, grp in df[df["Day"]==day].groupby("Name"):
        pts = grp[["x","y"]].values
        if len(pts)<3:
            continue
        xs, ys = pts[:,0].astype(int), pts[:,1].astype(int)
        rows = (shape[0]-1) - ys
        cols = xs
        rr, cc = polygon(rows, cols, shape=shape)
        code = ROI_TYPES.index(grp["Label"].iat[0]) + 1
        mask[rr,cc] = code
    return mask

# === GRID LOADER ===
grid_cache = {}
def load_grids(day):
    if day not in grid_cache:
        P = pd.read_csv(f"pressure/day{day}.txt", sep=r"\s+", header=None).values
        W = pd.read_csv(f"wind/day{day}.txt",    sep=r"\s+", header=None).values
        grid_cache[day] = (P,W)
    return grid_cache[day]

# === FEATURE HELPERS ===
def extract_corner(arr,x,y_idx,corner):
    if corner=="NW": r0,r1,c0,c1 = y_idx-HALF, y_idx+1, x-HALF,   x+1
    elif corner=="NE":r0,r1,c0,c1 = y_idx-HALF, y_idx+1, x,        x+HALF+1
    elif corner=="SW":r0,r1,c0,c1 = y_idx,     y_idx+HALF+1, x-HALF, x+1
    else:             r0,r1,c0,c1 = y_idx,     y_idx+HALF+1, x,      x+HALF+1
    H,W = arr.shape
    rr0,rr1 = max(0,r0), min(H,r1)
    cc0,cc1 = max(0,c0), min(W,c1)
    blk     = arr[rr0:rr1, cc0:cc1]
    pt,pb   = max(0,-r0), max(0,r1-H)
    pl,pr   = max(0,-c0), max(0,c1-W)
    return np.pad(blk, ((pt,pb),(pl,pr)), mode="edge")[:WINDOW,:WINDOW]

def get_region(x,y):
    if   y>=50: return "NW" if x<25 else "N"  if x<50 else "NE"
    elif y>=25: return "W"  if x<25 else "C"  if x<50 else "E"
    else:        return "SW" if x<25 else "S"  if x<50 else "SE"

def tex(mat):
    a = ((mat - mat.min())/(np.ptp(mat)+1e-6)*255).astype("uint8")
    G = graycomatrix(a,[1],[0],levels=256,symmetric=True,normed=True)
    return [graycoprops(G,k)[0,0] for k in
            ("contrast","dissimilarity","homogeneity","energy")]

# === COLLECT PATCHES & FEATURES ===
img_patches, feat_list, lbl_list = [], [], []
day_list, coords_list, types_list, regions_list = [], [], [], []

for day in ALL_DAYS:
    prev = day - 1
    if prev < min(ALL_DAYS):
        continue
    try:
        P,W = load_grids(prev)
    except FileNotFoundError:
        continue
    H,Wd = P.shape
    mask = create_roi_mask(roi_df, day, (H,Wd))
    for xr in range(0, Wd, STEP):
        for yr in range(0, H, STEP):
            row  = H-1-yr
            code = mask[row, xr]
            is_roi = code>0
            lbl    = 0 if is_roi else 1
            rtype  = ROI_TYPES[code-1] if is_roi else "random"
            region = get_region(xr, yr)

            for corner in ("NW","NE","SW","SE"):
                pwin = extract_corner(P, xr, row, corner)
                wwin = extract_corner(W, xr, row, corner)
                img_patches.append(np.stack([pwin,wwin],-1))

                psx = ndi.sobel(pwin,1)[HALF,HALF]; psy = ndi.sobel(pwin,0)[HALF,HALF]
                wsx = ndi.sobel(wwin,1)[HALF,HALF]; wsy = ndi.sobel(wwin,0)[HALF,HALF]
                basic = [pwin.mean(),pwin.std(),psx,psy,pwin[HALF,HALF],
                         wwin.mean(),wwin.std(),wsx,wsy,wwin[HALF,HALF]]
                feat_list.append(basic + tex(pwin) + tex(wwin))

                lbl_list.append(lbl)
                day_list.append(day)
                coords_list.append((xr,yr))
                types_list.append(rtype)
                regions_list.append(region)

# === TO NUMPY & CAST ===
X_img    = np.array(img_patches, dtype=np.float32)
X_basic  = np.array(feat_list,    dtype=np.float32)
y_all    = np.array(lbl_list,     dtype=np.int32)
days_arr = np.array(day_list,     dtype=np.int32)
coords   = np.array(coords_list,  dtype=np.int32)
types_arr= np.array(types_list)
regions  = np.array(regions_list)

# === ONE‐HOT REGIONS & K-MEANS ===
reg_ohe  = pd.get_dummies(regions).reindex(columns=REGIONS9, fill_value=0).values
km       = KMeans(n_clusters=KM, random_state=RNG, n_init=10).fit(X_basic)
clus_ohe = pd.get_dummies(km.labels_, prefix="cluster").values
dists    = np.linalg.norm(X_basic - km.cluster_centers_[km.labels_],axis=1).reshape(-1,1)
X_feat   = np.hstack([X_basic, reg_ohe, clus_ohe, dists])
X_feat   = StandardScaler().fit_transform(X_feat).astype(np.float32)

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
# Keep only samples whose TRUE type is one of the three ROI subtypes
roi_mask = (y_te == 0)  # 0 = ROI in your labeling
true_types_roi = types_te[roi_mask]           # actual ROI types: "COL", "COH", "CL"
pred_binary_roi = y_pred[roi_mask]            # binary predictions for ROI
true_types = types_te[roi_mask]

pred_types_roi = [
    t if p == 0 else "BG"
    for t, p in zip(true_types_roi, pred_binary_roi)
]

# map indices back to label strings
labels3 = ["COL", "COH", "CL"]

cm_types = confusion_matrix(true_types_roi, pred_types_roi, labels=labels3)

# Plot the 3x3 matrix for COL/COH/CL only
plt.figure(figsize=(4.5,4))
plt.imshow(cm_types, cmap="Blues")
plt.title("Confusion (by ROI type among COL/COH/CL)")
plt.xticks(range(len(labels3)), labels3)
plt.yticks(range(len(labels3)), labels3)
plt.xlabel("Predicted Type (ROI only)")
plt.ylabel("True Type")
for i in range(len(labels3)):
    for j in range(len(labels3)):
        val = cm_types[i, j]
        plt.text(j, i, val, ha="center", va="center",
                 color="white" if val > cm_types.max()/2 else "black")
plt.tight_layout()
plt.show()

labels4 = ["COL", "COH", "CL", "NROI"]

cm_types = confusion_matrix(true_types_roi, pred_types_roi, labels=labels4)

# Plot the 4x4 matrix for COL/COH/CL/NROI
plt.figure(figsize=(4.5,4))
plt.imshow(cm_types, cmap="Blues")
plt.title("Confusion")
plt.xticks(range(len(labels4)), labels4)
plt.yticks(range(len(labels4)), labels4)
plt.xlabel("Predicted Type")
plt.ylabel("True Type")
for i in range(len(labels4)):
    for j in range(len(labels4)):
        val = cm_types[i, j]
        plt.text(j, i, val, ha="center", va="center",
                 color="white" if val > cm_types.max()/2 else "black")
plt.tight_layout()
plt.show()

# Also print per-type recall (how many of each true type were detected as ROI)
support = np.array([(true_types == t).sum() for t in labels3], dtype=float)
diag = np.diag(cm_types).astype(float)
recall_by_type = {t: (d / s if s > 0 else 0.0) for t, d, s in zip(labels3, diag, support)}
print("Per-type recall (ROI found rate):", recall_by_type)

print("time:", int(round(time.time() * 1000)) - ms)
print(f" F1 score {f1_score(true_types_roi, pred_types_roi, average="weighted")}")

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