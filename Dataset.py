import pandas as pd
import pickle

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

features = db[["RID", "PTID"]]

inp = input().split()
feats_to_add = []
for feat in inp:
    feats_to_add.append("\'" + feat + "\'")

print(feats_to_add)

with open('feats.pickle', 'wb') as handle:
    pickle.dump(feats_to_add, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('feats.pickle', 'rb') as handle:
    feats = pickle.load(handle)

