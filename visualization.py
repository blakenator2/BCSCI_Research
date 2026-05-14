import matplotlib.pyplot as plt
import matplotlib.patches    as mpatches
import matplotlib.colors     as mcolors
from skimage.draw    import polygon
from scipy           import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
import numpy as np
import pickle

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

cmap_types = mcolors.ListedColormap(["lightgrey"] + [TYPE_COLORS[t] for t in ROI_TYPES])
type_code  = {t:i+1 for i,t in enumerate(ROI_TYPES)}

roi_df = pd.read_csv("roi_data_with_status.csv")
required = {"Day","Name","x","y","Label"}
if not required.issubset(roi_df.columns):
    raise RuntimeError(f"roi_data_with_status.csv must contain columns: {required}")

with open('train_patches.pkl', 'rb') as f:
        t_data = pickle.load(f)

img_patches = t_data['img_patches']
X_basic = t_data['X_basic']
y_all = t_data['y_all']
coords = t_data['coords']
types_arr = t_data['types_arr']
regions_codes = t_data['regions_codes']
days_arr = t_data['days_arr']


for day in TRAIN_DAYS:
    day_mask = days_arr == day
    if not day_mask.any():
        continue

    P, _       = load_grids(day - 1)
    H, W       = P.shape
    true_mask  = create_roi_mask(roi_df, day, (H, W))

    coords_day = coords[day_mask]
    types_day  = types_arr[day_mask]
    preds_day  = y_all[day_mask]

    pred_mask = np.zeros((H, W), dtype=int)
    for (x, y), tp, pr in zip(coords_day, types_day, preds_day):
        pred_type = label_to_type[pr]
        if (pr >= 0 and pr <= 3) and tp in type_code:
            pred_mask[H-1-y, x] = type_code[pred_type]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(pred_mask, cmap=cmap_types, origin="upper", vmin=0, vmax=len(ROI_TYPES))
    ax.set_title(f"Mask — Day {day}"); ax.axis("off")
    patches = [mpatches.Patch(color=cmap_types(i), label=l)
               for i, l in enumerate(["BG"] + ROI_TYPES)]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left", title="ROI Type")
    plt.tight_layout()
    plt.show()