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
DIR = "/home/abebe9849/Nploid/src/outputs/2023-01-08/HIPT_unfreeze2_aug"

folds =pd.read_csv(f"{DIR}/oof_nosiy_.csv")

print(folds.columns)
fold = 4
#exit()
val_idx = folds[folds['fold'] == fold].index
val_folds = folds.loc[val_idx].reset_index(drop=True)
import time
s = time.time()
RANDOM_STATE = 2023
perplexity = 10
clist = ["orange","pink","blue","brown","red","grey","yellow","green"]
clist = ["orange","blue"]
cdict = dict(zip(range(val_folds["label"].nunique()),clist))
color_dict ={"orange":[255,165,0],"pink":[255,10,255],"blue":[0,0,255],
"brown":[153,51,0],"red":[255,0,0],"grey":[150,150,150],
"yellow":[255,255,0],"green":[0,125,0],"black":[0,0,0],
}
features = cp.load(f"{DIR}/fold_{fold}_emb.npy")

#test_f = cp.load("/home/abe/kuma-ssl/dino/exp002/exp002-embed/test_embed_300e_vit_s.npy")

#cat_f = cp.concatenate([features,test_f])
X_reduced = TSNE(n_components=2, random_state=RANDOM_STATE,perplexity=perplexity).fit_transform(features)

cp.save(f"{DIR}/fold{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}.npy",X_reduced)

X_reduced = cp.asnumpy(X_reduced)
val_folds["tsne0"] = X_reduced[:, 0]
val_folds["tsne1"] = X_reduced[:, 1]

plt.figure(figsize=(15,15))

for i in [0,1]:
    tmp = val_folds[val_folds["label"]==i]
    if i==0:
        cls_name = "normal"
    else:
        cls_name = "Polyploid"
        
        
    x = tmp["tsne0"]
    y = tmp["tsne1"]
    plt.scatter(x,y, color=cdict[i], label=cls_name, s=25)
    
plt.legend()
plt.savefig(f"{DIR}/fold{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}.png")