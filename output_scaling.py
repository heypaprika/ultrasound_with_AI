import numpy as np
from sklearn.preprocessing import minmax_scale

X = np.array([[ 0.00001, 0.000012, 0.0004],[ 0.0002, 0.003, 0.00014],[ 0.000032, 0.0002001, 0.0000001]])
X_MinMax_scaled = minmax_scale(X, axis=0, copy=True)
print(X_MinMax_scaled)