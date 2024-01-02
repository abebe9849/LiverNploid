

import cupy as cp

from cuml import TSNE
from cuml import KMeans
from cuml.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


#from cuml import UMAP
#from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
DIR = "/home/abebe9849/Nploid/src/outputs/2023-01-06/_wo_ssr_add_brightcont/"
DIR = "/home/abebe9849/Nploid/src/outputs/2023-01-07/b0_aug"
DIR = "/home/abebe9849/Nploid/src/outputs/2023-01-30/b0_aug_gray"

DIR = "/home/abebe9849/Nploid/src/outputs/2023-03-12/HIPT_unfreeze1_aug_select"
test =pd.read_csv(f"{DIR}/test.csv")
df = pd.read_csv("/home/abebe9849/Nploid/新岡先生共有_臨床情報.csv")


th = 1
dict_ = dict(zip(df["AI No."],df["OS"]))
dict_2= dict(zip(df["AI No."],df["観察"]))

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

test["survival"]=test["WSI_ID"].apply(func)
test["live_dead"]=test["WSI_ID"].apply(func2)

test = test.dropna(subset=["live_dead"]).reset_index(drop=True)

test["label"]=test[["survival","live_dead"]].apply(func3,axis=1)



fold = 4
#exit()

import time
s = time.time()
RANDOM_STATE = 2023
perplexity = 30
clist = ["orange","pink","blue","brown","red","grey","yellow","green"]
clist = ["orange","blue"]
cdict = dict(zip(range(test["label"].nunique()),clist))
color_dict ={"orange":[255,165,0],"pink":[255,10,255],"blue":[0,0,255],
"brown":[153,51,0],"red":[255,0,0],"grey":[150,150,150],
"yellow":[255,255,0],"green":[0,125,0],"black":[0,0,0],
}
#features = cp.load(f"{DIR}/test_{fold}_emb.npy")

#X_reduced = TSNE(n_components=2, random_state=RANDOM_STATE,perplexity=perplexity).fit_transform(features)

#cp.save(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}.npy",X_reduced)

#X_reduced = cp.asnumpy(X_reduced)
X_reduced = np.load(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}.npy")
test["tsne0"] = X_reduced[:, 0]
test["tsne1"] = X_reduced[:, 1]

plt.figure(figsize=(15,15))

for i in [0,1]:
    tmp = test[test["label"]==i]
    if i==0:
        cls_name = "survival"
    else:
        cls_name = "dead"
        
        
    x = tmp["tsne0"]
    y = tmp["tsne1"]
    plt.scatter(x,y, color=cdict[i], label=cls_name, s=20)
    
plt.legend()
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_y{th}.png",dpi=300)
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_y{th}.tiff",dpi=300)



