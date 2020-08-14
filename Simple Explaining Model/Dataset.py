import pandas as pd
import pickle
import numpy as np
import torch

# Loading DataSet
db = pd.read_csv('Pickles/tadpole-preprocessed - Tadpole dataset - Sheet1.csv')

y_pred = db["DX_bl"].to_numpy()

to_add = 0
add_dict = dict()

# Converting predictions to numbers
for i in range(len(y_pred)):
    if y_pred[i] not in add_dict:
        add_dict[y_pred[i]] = to_add
        to_add += 1
    y_pred[i] = add_dict[y_pred[i]]

# Getting feature names
inp = input().split()
feats_to_add = []
for feat in inp:
    feats_to_add.append("\'" + feat + "\'")

# Removing rows containing Empty values
feats_to_add = np.delete(feats_to_add, [113, 202, 222, 286, 313, 393], axis=0)

features = db[feats_to_add].to_numpy()

# Creating edges
edge_features = db["APOE4"].to_numpy()
adj = np.zeros((edge_features.shape[0], edge_features.shape[0]))
print(adj[0].shape)

for i in range(edge_features.shape[0]):
    for j in range(edge_features.shape[0]):
        if i != j:
            if edge_features[i] == edge_features[j]:
                adj[i, j] = 1

# Saving data
with open('Pickles/apoe_adj.pickle', 'wb') as handle:
    pickle.dump(adj, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('Pickles/feats.pickle', 'wb') as handle:
    pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('Pickles/preds.pickle', 'wb') as handle:
    pickle.dump(y_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
