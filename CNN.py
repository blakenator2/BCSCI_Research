import numpy as np
import pandas as pd
from skimage.draw    import polygon
from sklearn.cluster         import KMeans
from sklearn.preprocessing   import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle
from imblearn.under_sampling import RandomUnderSampler
import h5py
import math

import tensorflow as tf
from tensorflow.keras        import Model, Input, layers
import matplotlib.colors     as mcolors

WINDOW, HALF = 25, 24
TRAIN_DAYS = (
    list(range(1230,1300)) 
)
TEST_DAYS  = (
    list(range(1301, 1330)) 
)
ALL_DAYS   = TRAIN_DAYS + TEST_DAYS
RNG        = 42
KM         = 10
EPOCHS     = 40
BATCH      = 256
META_FILE  = "patches_data.npz"
IMG_FILE   = "patches_imgs.h5"
_SPLIT_KEYS = ["X_basic", "y_all", "coords", "types_arr", "regions_codes", "days_arr"]
REGIONS9   = ["NW","N","NE","W","C","E","SW","S","SE"]
ROI_TYPES  = ["COL","CL","COH","NROI"]
TYPE_COLORS = {"COL":"red","CL":"blue","COH":"green","NROI":"orange"}

class PatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, h5_path, split, days_arr, day_list, X_feat, y, batch_size=256):
        super().__init__()
        self.h5_path    = h5_path
        self.split      = split
        self.X_feat     = X_feat
        self.y          = y
        self.batch_size = batch_size

        idx = np.where(np.isin(days_arr, list(day_list)))[0]
        self.start = int(idx[0])
        self.end   = int(idx[-1]) + 1
        self.n     = self.end - self.start

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, i):
        bs    = self.batch_size
        lo    = self.start + i * bs
        hi    = min(self.start + (i + 1) * bs, self.end)
        local = slice(lo - self.start, hi - self.start)

        with h5py.File(self.h5_path, "r") as h5:
            patches = h5[f"{self.split}_img_patches"][lo:hi]

        return (
            (patches, self.X_feat[local]),  # tuple not list
            self.y[local]
        )

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

type_to_label = {"COL": 0, "CL": 1, "COH": 2, "NROI": 3, "BG": 4}
label_to_type = {v: k for k, v in type_to_label.items()}

cmap_types = mcolors.ListedColormap(["lightgrey"] + [TYPE_COLORS[t] for t in ROI_TYPES])
type_code  = {t:i+1 for i,t in enumerate(ROI_TYPES)}

roi_df = pd.read_csv("roi_data_with_status.csv")
required = {"Day","Name","x","y","Label"}
if not required.issubset(roi_df.columns):
    raise RuntimeError(f"roi_data_with_status.csv must contain columns: {required}")


raw = np.load(META_FILE, allow_pickle=True)
tr_data = {k: raw[f"tr_{k}"] for k in _SPLIT_KEYS}
te_data = {k: raw[f"te_{k}"] for k in _SPLIT_KEYS}

X_basic     = np.vstack([tr_data['X_basic'], te_data['X_basic']])
y_all       = np.concatenate([tr_data['y_all'], te_data['y_all']])
coords      = np.vstack([tr_data['coords'], te_data['coords']])
types_arr   = np.concatenate([tr_data['types_arr'], te_data['types_arr']])
regions_codes = np.concatenate([tr_data['regions_codes'], te_data['regions_codes']])
days_arr    = np.concatenate([tr_data['days_arr'], te_data['days_arr']])

# Convert region codes to one-hot
reg_ohe = np.zeros((len(regions_codes), 9), dtype=np.float32)
reg_ohe[np.arange(len(regions_codes)), regions_codes] = 1

# Rebuild regions array for compatibility
regions = np.array([REGIONS9[c] for c in regions_codes])

# === K-MEANS ===
train_mask = np.isin(days_arr, TRAIN_DAYS)
test_mask  = np.isin(days_arr, TEST_DAYS)

print("Running K-means clustering...")
# km = KMeans(n_clusters=KM, random_state=RNG, n_init=10).fit(X_basic)
# clus_ohe = np.zeros((len(X_basic), KM), dtype=np.float32)
# clus_ohe[np.arange(len(X_basic)), km.labels_] = 1
# dists = np.linalg.norm(X_basic - km.cluster_centers_[km.labels_], axis=1).reshape(-1, 1)

# X_feat = np.hstack([X_basic, reg_ohe, clus_ohe, dists])
# np.savez_compressed('X_feat.npz', x=X_feat)

X_feat = np.load('X_feat.npz')['x']

scaler = StandardScaler()
X_feat_tr = scaler.fit_transform(X_feat[train_mask])
X_feat_te = scaler.transform(X_feat[test_mask])

# === SPLIT BY DAY === 
y_tr,      y_te       = y_all[train_mask],  y_all[test_mask]
coords_tr, coords_te  = coords [train_mask], coords [test_mask]
types_tr,  types_te   = types_arr[train_mask], types_arr[test_mask]
days_tr,   days_te    = days_arr[train_mask], days_arr[test_mask]

print("Train class counts:", np.bincount(y_tr))
print("Test  class counts:", np.bincount(y_te))

# === ROBUST CLASS WEIGHTS ===
classes_present = np.unique(y_tr)

print("Classes present in y_tr:", classes_present)

# === EARLY STOPPING FUNCTION ===
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',   
    patience=3,           
    restore_best_weights=True,
    start_from_epoch=5
)

# === BUILD & TRAIN CNN ===

#Make CM ROI based not per pixel
patch_input = Input(shape=(25, 25, 2), name="patch_input")
feat_input  = Input(shape=(X_feat_tr.shape[1],), name="feat_input")

x = layers.Conv2D(32, 3, activation="relu", padding="same")(patch_input)
x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)

x = layers.Flatten()(x)
x = layers.Dense(64, activation="relu")(x)
y = layers.Dropout(0.4)(x)
y = layers.Dense(64, activation="relu")(feat_input)
y = layers.Dropout(0.4)(y)
y = layers.Dense(32, activation="relu")(y)
y = layers.Dropout(0.4)(y)

combined = layers.concatenate([x, y])
combined = layers.Dense(128, activation="relu")(combined)
combined = layers.Dropout(0.4)(combined)
combined = layers.Dense(64, activation="relu")(combined)
combined = layers.Dropout(0.4)(combined)
output   = layers.Dense(5, activation="softmax")(combined)

cnn = Model(inputs=[patch_input, feat_input], outputs=output)

cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

train_gen = PatchGenerator(IMG_FILE, "tr", tr_data['days_arr'], TRAIN_DAYS, X_feat_tr, y_tr, BATCH)
test_gen  = PatchGenerator(IMG_FILE, "tr", tr_data['days_arr'], TEST_DAYS,  X_feat_te, y_te, BATCH)

history = cnn.fit(
    train_gen,
    validation_data=test_gen,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    verbose=1
)

cnn.save("CNN.keras")
print('Saved CNN')

# Evaluate on test set
loss, acc = cnn.evaluate(test_gen, verbose=0)
print(f"Test accuracy: {acc:.3f}")

y_type_tr = np.array([type_to_label[t] for t in types_tr])
y_type_te = np.array([type_to_label[t] for t in types_te])

print(f"ROI training samples: {len(y_type_tr)}")
print(f"ROI test samples: {len(y_type_te)}")
print("Type distribution (train):", {label_to_type[i]: (y_type_tr==i).sum() for i in range(5)})
print("Type distribution (test):", {label_to_type[i]: (y_type_te==i).sum() for i in range(5)})