import pandas as pd
import pickle
import numpy as np

db = pd.read_csv('tadpole-preprocessed - Tadpole dataset - Sheet1.csv')

# print(db[["RID","PTID"]])

y_pred = db["DX_bl"].to_numpy()

to_add = 0
add_dict = dict()

for i in range(len(y_pred)):
    if y_pred[i] not in add_dict:
        add_dict[y_pred[i]] = to_add
        to_add += 1
    y_pred[i] = add_dict[y_pred[i]]

# with open('feats.pickle', 'rb') as handle:
#     feats = pickle.load(handle)

# with open('feats.pickle', 'wb') as handle:
#     pickle.dump(np.expand_dims(features,axis=0), handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('preds.pickle', 'wb') as handle:
#     pickle.dump(np.expand_dims(y_pred,axis=0), handle, protocol=pickle.HIGHEST_PROTOCOL)

age = db["AGE"].to_numpy()
adj = np.zeros((age.shape[0], age.shape[0]))
print(adj[0].shape)

for i in range(age.shape[0]):
    for j in range(age.shape[0]):
        if i != j:
            if abs(age[i] - age[j]) <= 2:
                adj[i, j] = 1

with open('age_adj.pickle', 'wb') as handle:
    pickle.dump(adj, handle, protocol=pickle.HIGHEST_PROTOCOL)
