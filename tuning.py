import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing   import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from tqdm import tqdm
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print(f"GPUs available: {torch.cuda.device_count()}")  # should print 2

TRAIN_DAYS = (
    list(range(0,    366))  +  # 2000
    list(range(366,  731))  +  # 2001
    list(range(731,  1096))
)
TEST_DAYS  = (
    list(range(2557, 2922)) +  # 2007
    list(range(2922, 3288)) 
)
ALL_DAYS   = TRAIN_DAYS + TEST_DAYS
META_FILE  = "patches_data.npz"
IMG_FILE   = "patches_imgs.h5"
_SPLIT_KEYS = ["X_basic", "y_all", "coords", "types_arr", "regions_codes", "days_arr"]

X_feat = np.load('X_feat.npz')['x']

raw = np.load(META_FILE, allow_pickle=True)
tr_data = {k: raw[f"tr_{k}"] for k in _SPLIT_KEYS}
te_data = {k: raw[f"te_{k}"] for k in _SPLIT_KEYS}

X_basic     = np.vstack([tr_data['X_basic'], te_data['X_basic']])
y_all       = np.concatenate([tr_data['y_all'], te_data['y_all']])
coords      = np.vstack([tr_data['coords'], te_data['coords']])
types_arr   = np.concatenate([tr_data['types_arr'], te_data['types_arr']])
regions_codes = np.concatenate([tr_data['regions_codes'], te_data['regions_codes']])
days_arr    = np.concatenate([tr_data['days_arr'], te_data['days_arr']])
days_arr_te = tr_data['days_arr']

train_mask = np.isin(days_arr, TRAIN_DAYS)
test_mask  = np.isin(days_arr, TEST_DAYS) 

scaler = StandardScaler()
X_feat_tr = scaler.fit_transform(X_feat[train_mask])
X_feat_te = scaler.transform(X_feat[test_mask])

X_feat_tr, X_feat_te  = X_feat[train_mask], X_feat[test_mask]
y_tr,      y_te       = y_all[train_mask],  y_all[test_mask]
coords_tr, coords_te  = coords [train_mask], coords [test_mask]
types_tr,  types_te   = types_arr[train_mask], types_arr[test_mask]
days_tr,   days_te    = days_arr[train_mask], days_arr[test_mask]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(y_tr), 
    y=y_tr
)
weights[3]=0
class_weight_dict = dict(zip(np.unique(y_tr), weights))

class_weights = torch.tensor(
    [class_weight_dict[i] for i in range(5)],  # assumes classes 0-4
    dtype=torch.float32
).to(device)

# Convert arrays to tensors
X_tr_t = torch.tensor(X_feat_tr, dtype=torch.float32).to(device)
y_tr_t  = torch.tensor(y_tr,      dtype=torch.long).to(device)
X_te_t  = torch.tensor(X_feat_te, dtype=torch.float32).to(device)
y_te_t  = torch.tensor(y_te,      dtype=torch.long).to(device)

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=256, shuffle=True)

def build_model(trial):
    activation_name = trial.suggest_categorical("activation", ["relu", "sigmoid"])
    activation_map  = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid}
    act             = activation_map[activation_name]

    input_size = X_feat_tr.shape[1]
    layers = []
    prev_units = input_size

    for i in range(1, 4):  # 3 hidden layers
        units = trial.suggest_int(f"units{i}", 32, 512, step=32)
        layers.append(nn.Linear(prev_units, units))
        layers.append(act())
        if trial.suggest_categorical(f"dropout{i}", [True, False]):
            layers.append(nn.Dropout(0.25))
        prev_units = units

    layers.append(nn.Linear(prev_units, 5))  # output layer (5 classes)
    return nn.Sequential(*layers).to(device)

def objective(trial):
    torch.cuda.empty_cache()
    model     = build_model(trial)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(8):
        model.train()
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/8"):
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_te_t), y_te_t).item()

    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=3)

print("Best params:", study.best_params)

# Rebuild the best model architecture
best_model = build_model(study.best_trial)

#https://keras.io/keras_tuner/getting_started/
# Value             |Best Value So Far |Hyperparameter
# 288               |288               |units1
# sigmoid           |sigmoid           |activation
# False             |False             |dropout
# 448               |448               |units2
# 256               |256               |units3