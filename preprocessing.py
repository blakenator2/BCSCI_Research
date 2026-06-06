from skimage.draw    import polygon
from scipy           import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops
from numba import jit
from multiprocessing import Pool, cpu_count
from functools import partial
import pandas as pd
import numpy as np
import time
import shutil
import h5py

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
TRAIN_SET  = set(TRAIN_DAYS)
STEP       = 1
RND_WEIGHT = 2.0
REGIONS9   = ["NW","N","NE","W","C","E","SW","S","SE"]
ROI_TYPES  = ["COL","CL","COH","NROI"]
TYPE_COLORS = {"COL":"red","CL":"blue","COH":"green","NROI":"orange"}

type_to_label = {"COL": 0, "CL": 1, "COH": 2, "NROI": 3, "random": 4}
label_to_type = {v: k for k, v in type_to_label.items()}

META_FILE  = "patches_data.npz"
IMG_FILE   = "patches_imgs.h5"
FLUSH_EVERY = 50

_SPLIT_KEYS = ["X_basic", "y_all", "coords", "types_arr", "regions_codes", "days_arr"]


def _h5_init(path):
    """Create the HDF5 file with resizable datasets for tr and te img_patches."""
    with h5py.File(path, "w") as f:
        for split in ("tr", "te"):
            # maxshape=(None, …) makes the first axis unlimited
            f.create_dataset(
                f"{split}_img_patches",
                shape=(0, WINDOW, WINDOW, 2),
                maxshape=(None, WINDOW, WINDOW, 2),
                dtype=np.float32,
                chunks=(256, WINDOW, WINDOW, 2),   # chunk ≈ 1 MB
                compression="gzip",
                compression_opts=4,
            )


def _h5_append(path, split, patches):
    """Append a (N, 25, 25, 2) array to the appropriate dataset."""
    if len(patches) == 0:
        return
    with h5py.File(path, "a") as f:
        ds = f[f"{split}_img_patches"]
        old_len = ds.shape[0]
        new_len = old_len + len(patches)
        ds.resize(new_len, axis=0)
        ds[old_len:new_len] = patches


def _empty_split():
    return {k: [] for k in _SPLIT_KEYS}


def _flush_meta(tr_buf, te_buf, out_file):
    """Concatenate metadata buffers and write compressed npz atomically."""

    def concat(buf, key):
        arrs = buf[key]
        if not arrs:
            return np.array([])
        if key in ("X_basic", "coords"):
            return np.vstack(arrs)
        return np.concatenate(arrs)

    save_dict = {}
    for key in _SPLIT_KEYS:
        save_dict[f"tr_{key}"] = concat(tr_buf, key)
        save_dict[f"te_{key}"] = concat(te_buf, key)

    tmp = out_file + ".tmp"
    np.savez_compressed(tmp, **save_dict)
    src = tmp if tmp.endswith(".npz") else tmp + ".npz"
    shutil.move(src, out_file)


def _accumulate(buf, result):
    """Append one day's metadata result into a split buffer."""
    buf["X_basic"].append(result["features"])
    buf["y_all"].append(result["labels"])
    buf["coords"].append(result["coords"])
    buf["types_arr"].append(result["types"])
    buf["regions_codes"].append(result["regions"])
    buf["days_arr"].append(
        np.full(len(result["labels"]), result["day"], dtype=np.int32)
    )


def create_roi_mask(df, day, shape):
    mask = np.zeros(shape, dtype=np.int8)
    day_data = df[df["Day"] == day]
    for name, grp in day_data.groupby("Name"):
        pts = grp[["x", "y"]].values
        xs, ys = pts[:, 0].astype(np.int32), pts[:, 1].astype(np.int32)
        rows = (shape[0] - 1) - ys
        cols = xs
        rr, cc = polygon(rows, cols, shape=shape)
        code = ROI_TYPES.index(grp["Label"].iat[0]) + 1
        mask[rr, cc] = code
    return mask


def load_grids(day):
    P = pd.read_csv(f"pressure/day{day}.txt", sep=r"\s+", header=None).values.astype(np.float32)
    W = pd.read_csv(f"wind/day{day}.txt",     sep=r"\s+", header=None).values.astype(np.float32)
    return P, W


@jit(nopython=True, cache=True)
def extract_window_safe(arr, x, y_idx, corner_idx, H, W):
    if corner_idx == 0:
        r0, r1, c0, c1 = y_idx - HALF, y_idx + 1, x - HALF, x + 1
    elif corner_idx == 1:
        r0, r1, c0, c1 = y_idx - HALF, y_idx + 1, x, x + HALF + 1
    elif corner_idx == 2:
        r0, r1, c0, c1 = y_idx, y_idx + HALF + 1, x - HALF, x + 1
    else:
        r0, r1, c0, c1 = y_idx, y_idx + HALF + 1, x, x + HALF + 1

    rr0, rr1 = max(0, r0), min(H, r1)
    cc0, cc1 = max(0, c0), min(W, c1)
    result    = np.zeros((WINDOW, WINDOW), dtype=np.float32)
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
                result[i, j] = result[result_r1 - 1, j]
    if result_c0 > 0:
        for i in range(WINDOW):
            for j in range(result_c0):
                result[i, j] = result[i, result_c0]
    if result_c1 < WINDOW:
        for i in range(WINDOW):
            for j in range(result_c1, WINDOW):
                result[i, j] = result[i, result_c1 - 1]

    return result


@jit(nopython=True, cache=True)
def compute_basic_features(pwin, wwin, psx, psy, wsx, wsy):
    pmean   = np.mean(pwin)  
    pstd = np.std(pwin)  
    pcenter = pwin[HALF, HALF]

    wmean = np.mean(wwin)  
    wstd = np.std(wwin)
    wcenter = wwin[HALF, HALF]

    return np.array([pmean, pstd, psx, psy, pcenter,
                     wmean, wstd, wsx, wsy, wcenter], dtype=np.float32)


def tex_fast(mat):
    mat_range = mat.max() - mat.min()
    if mat_range < 1e-6:
        return [0.0, 0.0, 1.0, 1.0]
    a = ((mat - mat.min()) / mat_range * 255).astype(np.uint8)
    G = graycomatrix(a, [1], [0], levels=256, symmetric=True, normed=True)
    return [
        graycoprops(G, 'contrast')[0, 0],
        graycoprops(G, 'dissimilarity')[0, 0],
        graycoprops(G, 'homogeneity')[0, 0],
        graycoprops(G, 'energy')[0, 0],
    ]


@jit(nopython=True, cache=True)
def get_region_code(x, y):
    if y >= 50:
        if x < 25: return 0
        elif x < 50: return 1
        else: return 2
    elif y >= 25:
        if x < 25: return 3
        elif x < 50: return 4
        else: return 5
    else:
        if x < 25: return 6
        elif x < 50: return 7
        else: return 8


def process_day(day, roi_df_subset):
    prev = day - 1
    try:
        P, W = load_grids(prev)
    except FileNotFoundError:
        return None

    H, Wd = P.shape
    mask  = create_roi_mask(roi_df_subset, day, (H, Wd))

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 8.0
    P_sx = ndi.convolve(P, sobel_x, mode='constant')
    P_sy = ndi.convolve(P, sobel_y, mode='constant')
    W_sx = ndi.convolve(W, sobel_x, mode='constant')
    W_sy = ndi.convolve(W, sobel_y, mode='constant')

    patches_list  = []
    feat_list     = []
    lbl_list      = []
    coords_list   = []
    types_list    = []
    regions_list  = []

    for xr in np.arange(0, Wd, STEP):
        for yr in np.arange(0, H, STEP):
            row    = H - 1 - yr
            code   = mask[row, xr]
            is_roi = code > 0
            lbl    = type_to_label[ROI_TYPES[code - 1]] if is_roi else 4
            rtype  = ROI_TYPES[code - 1] if is_roi else "BG"
            psx, psy = P_sx[row, xr], P_sy[row, xr]
            wsx, wsy = W_sx[row, xr], W_sy[row, xr]

            for corner_idx in range(4):
                pwin = extract_window_safe(P, xr, row, corner_idx, H, Wd)
                wwin = extract_window_safe(W, xr, row, corner_idx, H, Wd)

                patches_list.append(np.stack([pwin, wwin], axis=-1))  

                basic = compute_basic_features(pwin, wwin, psx, psy, wsx, wsy)
                feat_list.append(np.concatenate([basic, tex_fast(pwin), tex_fast(wwin)]))

                lbl_list.append(lbl)
                coords_list.append((xr, yr))
                types_list.append(rtype)
                regions_list.append(get_region_code(xr, yr))

    return {
        'day':      day,
        'img_patches': np.array(patches_list,  dtype=np.float32), 
        'features':    np.array(feat_list,     dtype=np.float32),
        'labels':      np.array(lbl_list,      dtype=np.int32),
        'coords':      np.array(coords_list,   dtype=np.int32),
        'types':       np.array(types_list),
        'regions':     np.array(regions_list,  dtype=np.int32),
    }

def load_data(meta_path=META_FILE, img_path=IMG_FILE):
    raw = np.load(meta_path, allow_pickle=True)
    tr_data = {k: raw[f"tr_{k}"] for k in _SPLIT_KEYS}
    te_data = {k: raw[f"te_{k}"] for k in _SPLIT_KEYS}

    h5 = h5py.File(img_path, "r")       
    tr_data["img_patches"] = h5["tr_img_patches"]
    te_data["img_patches"] = h5["te_img_patches"]

    return tr_data, te_data, h5   # return h5 so caller can close it


if __name__ == '__main__':
    NUM_WORKERS = max(1, cpu_count() - 1)
    print(f"Using {NUM_WORKERS} worker processes")

    roi_df = pd.read_csv("roi_data_with_status.csv")
    required = {"Day", "Name", "x", "y", "Label"}
    if not required.issubset(roi_df.columns):
        raise RuntimeError(f"roi_data_with_status.csv must contain columns: {required}")

    valid_days = [d for d in ALL_DAYS if d - 1 >= min(ALL_DAYS)]
    process_day_with_roi = partial(process_day, roi_df_subset=roi_df)

    # Initialise HDF5 file with empty resizable datasets
    _h5_init(IMG_FILE)

    print("Collecting patches and features in parallel...")
    start_time = time.time()

    tr_buf    = _empty_split()
    te_buf    = _empty_split()
    processed = 0

    with Pool(processes=NUM_WORKERS) as pool:
        for result in pool.imap(process_day_with_roi, valid_days):
            if result is None:
                continue

            split = "tr" if result['day'] in TRAIN_SET else "te"
            buf   = tr_buf if split == "tr" else te_buf

            _h5_append(IMG_FILE, split, result["img_patches"])

            _accumulate(buf, result)

            processed += 1

            if processed % FLUSH_EVERY == 0:
                _flush_meta(tr_buf, te_buf, META_FILE)
                elapsed = time.time() - start_time
                print(f"  [{processed}/{len(valid_days)}] flushed — "
                      f"{elapsed:.1f}s ({elapsed/processed:.2f}s/day)")

    # Final metadata flush
    _flush_meta(tr_buf, te_buf, META_FILE)
    print(f"\nDone. {processed} days in {time.time()-start_time:.1f}s")
    print(f"  Metadata : {META_FILE}")
    print(f"  Images   : {IMG_FILE}")