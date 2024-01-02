#conda activate rapids-0.19 

import cupy as cp

#from cuml import TSNE
from cuml import KMeans
from cuml.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

import cudf
#from cuml import UMAP
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
DIR = "/home/abebe9849/Nploid/src/outputs/2023-01-06/_wo_ssr_add_brightcont/"
DIR = "/home/abebe9849/Nploid/src/outputs/2023-01-07/b0_aug"
DIR = "/home/abebe9849/Nploid/src/outputs/2023-03-12/HIPT_unfreeze2_aug_select"

#DIR = "/home/abebe9849/Nploid/src/outputs/2023-01-08/HIPT_unfreeze1_aug"
test = pd.read_csv("/home/abebe9849/Nploid/src/outputs/SAVE3/HIPT_unfreeze1_aug/test_wo_duplicate.csv")

test_pred =pd.read_csv(f"{DIR}/test.csv")
df = pd.read_csv("/home/abebe9849/Nploid/新岡先生共有_臨床情報_松浦追記_20230131.csv")


th = 1
dict_ = dict(zip(df["AI No."],df["OS"]))
dict_2= dict(zip(df["AI No."],df["観察"]))


dict_PGCC = dict(zip(df["AI No."],df["PGCC_bi"]))
dict_histo = dict(zip(df["AI No."],df["組織型"]))


dict_pred = dict(zip(test_pred["file_path"],test_pred["pred"]))


dict_ = dict(zip(df["AI No."],df["heptitis_type"]))



def func(x):
    out = dict_[x]
    
    return out
def func2(x):
    outcome = dict_2[x]
    
    return outcome

def func3(x):
    if x[0]>=int(th*365):
        return 0
    else:
        if x[1]=="死亡":
            return 1
        else:
            return -1
        
def func(x):
    out = dict_[x]
    if out=="unknown":
        return None
    else:
        return out


V = set(dict_histo.values())
dict_histo_num = dict(zip(V,range(len(V))))
dict_num_histo = dict(zip(range(len(V)),V))

print(dict_histo_num)
def func1(x):
    out = dict_histo[x]
    out = dict_histo_num[out]
    if out=="unknown":
        return None
    else:
        return out
    
def func_dict_PGCC(x):
    out = dict_PGCC[x]
    return out

def func2(x):
    out = dict_pred[x]
    return out

#test["survival"]=test["patient"].apply(func)
#test["live_dead"]=test["patient"].apply(func2)

#test = test.dropna(subset=["live_dead"]).reset_index(drop=True)

#test["label"]=test[["survival","live_dead"]].apply(func3,axis=1)

test["PGCC"]=test["patient"].apply(func_dict_PGCC)


test["histo"]=test["patient"].apply(func1)


fold = 3
#exit()

import time
s = time.time()
RANDOM_STATE = 20230
perplexity = 30
clist = ["orange","pink","blue","brown","red","grey","yellow","green"]
clist = ["orange","blue"]
color_dict ={"orange":[255,165,0],"pink":[255,10,255],"blue":[0,0,255],
"brown":[153,51,0],"red":[255,0,0],"grey":[150,150,150],
"yellow":[255,255,0],"green":[0,125,0],"black":[0,0,0],
}

#test_df = cudf.read_csv(f"{DIR}/test.csv")
#"""
features =np.load(f"{DIR}/test_{fold}_emb.npy")


emb_s = []
for patient in test["patient"].unique():
    tmp =test_pred["WSI_ID"]==patient
    tmp =  test_pred[tmp].index.values
    emb_s.append(np.mean(features[tmp],axis=0))
    
features = np.stack(emb_s)
print(features.shape,test_pred.WSI_ID.nunique())


X_reduced = TSNE(n_components=2, random_state=RANDOM_STATE,perplexity=perplexity).fit_transform(features)

np.save(f"{DIR}/test_agg_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}.npy",X_reduced)


#X_reduced = np.load(f"{DIR}/test_agg_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}.npy")

#X_reduced = cp.asnumpy(X_reduced)

print(test.head())
test_g = pd.DataFrame(test.groupby("patient")[["pred","histo","PGCC"]].mean())
print(test_g.head())

test_g["tsne0"] = X_reduced[:, 0]
test_g["tsne1"] = X_reduced[:, 1]

plt.figure(figsize=(15,15))

clist = ["green","brown","lime","blue","red","grey"]

cdict = dict(zip(test_g["histo"].unique(),clist))

for label in test_g["histo"].unique():

    tmp = test_g[test_g["histo"]==label].reset_index(drop=True)
        
    x = tmp["tsne0"]
    y = tmp["tsne1"]
    plt.scatter(x,y, color=cdict[label], label=dict_num_histo[label], s=20)
    
plt.legend()
plt.savefig(f"{DIR}/test_agg_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_histo.tiff",dpi=300)
plt.savefig(f"{DIR}/test_agg_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_histo.png",dpi=300)


plt.figure(figsize=(15,15))
clist = ["blue","red","lime","blue","green","grey"]

cdict = dict(zip(test_g["PGCC"].unique(),clist))

for label in test_g["PGCC"].unique():

    tmp = test_g[test_g["PGCC"]==label].reset_index(drop=True)
    
    color = cdict[label]
    if label==1:
        label="PGCC"
    else:
        label="without PGCC"
        
    x = tmp["tsne0"]
    y = tmp["tsne1"]
    plt.scatter(x,y, color=color, label=label, s=20)
    
plt.legend()
plt.savefig(f"{DIR}/test_agg_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_PGCC.tiff",dpi=300)
plt.savefig(f"{DIR}/test_agg_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_PGCC.png",dpi=300)

"""
for i in [0,1]:
    tmp = test[test["label"]==i]
    if i==0:
        cls_name = "survival"
    else:
        cls_name = "dead"
        
        
    x = tmp["tsne0"]
    y = tmp["tsne1"]
    plt.scatter(x,y, color=cdict[i], label=cls_name, s=25)
    
plt.legend()
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_y{th}_agg.png")
"""


