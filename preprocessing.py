import matplotlib.pyplot as plt
from skimage.draw    import polygon
from scipy           import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster         import KMeans
from sklearn.preprocessing   import StandardScaler
from sklearn.utils           import class_weight
from sklearn.metrics         import classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from functools import partial
import pandas as pd
import numpy as np
import time
import pickle

#Lines 10817-10836 had CLOL not COL. Changed it but unsure of accuracy.

# === PARAMETERS (constants are safe) ===
WINDOW, HALF = 25, 24
TRAIN_DAYS = (
    list(range(0,    366))  +  # 2000
    list(range(366,  731))  +  # 2001
    list(range(731,  1096)) +  # 2002
    list(range(1096, 1461)) +  # 2003
    list(range(2192, 2557))    # 2006
)
TEST_DAYS  = (
    list(range(2557, 2922)) +  # 2007
    list(range(2922, 3288)) +  # 2008
    list(range(3288, 3651))    # 2009
)
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

type_to_label = {"COL": 0, "CL": 1, "COH": 2, "NROI": 3, "random": 4}
label_to_type = {v: k for k, v in type_to_label.items()}



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
            lbl = type_to_label[ROI_TYPES[code-1]] if is_roi else 4
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

# Calculate ms and NUM_WORKERS inside the main block
if __name__ == '__main__':
    ms = int(round(time.time() * 1000))

    # Load ROI data
    roi_df = pd.read_csv("roi_data_with_status.csv")
    required = {"Day","Name","x","y","Label"}
    if not required.issubset(roi_df.columns):
        raise RuntimeError(f"roi_data_with_status.csv must contain columns: {required}")

    # === PARALLEL COLLECTION ===
    print("Collecting patches and features...")
    start_time = time.time()

    # Filter days
    valid_days = [day for day in ALL_DAYS if day > 0]

    results = []
    for i, day in enumerate(valid_days):
        result = process_day(day, roi_df)
        if result is not None:
            results.append(result)
        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {i + 1}/{len(valid_days)} days in {elapsed:.1f}s ({elapsed/(i+1):.2f}s per day)")

    print(f"Processing complete in {time.time()-start_time:.1f}s")

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

    data = {
        'img_patches': img_patches,
        'X_basic': X_basic,
        'y_all': y_all,
        'coords': coords,
        'types_arr': types_arr,
        'regions_codes': regions_codes,
    }

    with open('arrays.pkl', 'wb') as f:
        pickle.dump(data, f)