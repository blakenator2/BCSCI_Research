import numpy as np

pred_types_strings = np.load("pred_types_strings.npy")
true_types_strings = np.load("true_types_strings.npy")
y_type_pred = np.load("y_type_pred.npy")

print("==================================")
print(y_type_pred.shape)
print("==================================")
print(pred_types_strings.shape)
print("==================================")
print(true_types_strings.shape)