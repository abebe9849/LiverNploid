

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

DIR = "/home/abebe9849/Nploid/src/outputs/2023-03-12/HIPT_unfreeze2_aug_select"

test = pd.read_csv("/home/abebe9849/Nploid/src/outputs/SAVE3/HIPT_unfreeze1_aug/test_wo_duplicate.csv")

test_pred =pd.read_csv(f"{DIR}/test.csv")
df = pd.read_csv("/home/abebe9849/Nploid/新岡先生共有_臨床情報_松浦追記_20230131.csv")



print(df["heptitis_type"].value_counts())


print(df["組織型"].value_counts())


df_new = pd.read_csv("/home/abebe9849/Nploid/20231003_heatmap_input.csv")





print(df.columns)

df_new["AI No."]=["AI"+i[2:] for i in df_new["HL_ID"].values]


#df = df[df["AI No."].isin(df_new["AI No."].unique())]



dict_histo = dict(zip(df["AI No."],df["組織型"]))

dict_PGCC = dict(zip(df_new["AI No."],df_new["PGCC_bi"]))


dict_pred = dict(zip(test_pred["file_path"],test_pred["pred"]))


dict_ = dict(zip(df_new["AI No."],df_new["heptitis_type"]))

print(df_new["heptitis_type"].value_counts())



def func(x):
    try:
        out = dict_[x]
    except:
        return None

    return out

def func1(x):
    out = dict_histo[x]
    if out=="unknown":
        return None
    else:
        return out
    
def func_dict_PGCC(x):
    try:
        out = dict_PGCC[x]
    except:
        return None
    return out

def func2(x):
    out = dict_pred[x]
    return out

test["label"]=test["patient"].apply(func)
test["PGCC"]=test["patient"].apply(func_dict_PGCC)



test["histo"]=test["patient"].apply(func1)

print(test["label"].unique())
fold = 4
#exit()

import time
s = time.time()
RANDOM_STATE = 2023
perplexity = 30

color_dict ={"orange":[255,165,0],"pink":[255,10,255],"blue":[0,0,255],
"brown":[153,51,0],"red":[255,0,0],"grey":[150,150,150],
"yellow":[255,255,0],"green":[0,255,0],"black":[0,0,0],"Lgreen":[0,125,0],
}
#features = cp.load(f"{DIR}/test_{fold}_emb.npy")
#print(features.shape,test.shape)
#X_reduced = TSNE(n_components=2, random_state=RANDOM_STATE,perplexity=perplexity).fit_transform(features)
#cp.save(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}.npy",X_reduced)

#X_reduced = cp.asnumpy(X_reduced)
X_reduced = np.load(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}.npy")

print(test.columns)



test["tsne0"] = X_reduced[:, 0]
test["tsne1"] = X_reduced[:, 1]

print(test.shape)
print(test["patient"].nunique())

test = test[test["patient"].isin(df_new["AI No."].unique())].reset_index(drop=True)

test = test[test["file_path"].isin(test_pred["file_path"].unique())].reset_index(drop=True)
print(test["patient"].nunique())

clist = ["blue","orange","gray","red","cyan","lime","deepskyblue","green"]
print(test["label"].unique())
cdict = dict(zip(test["label"].unique(),clist))

test["pred__"]=test["file_path"].apply(func2)

test = test.dropna(subset=["label"]).reset_index(drop=True)
#"""

plt.figure(figsize=(15,15))

for label in test["label"].unique():

    tmp = test[test["label"]==label].reset_index(drop=True)
        
    x = tmp["tsne0"]
    y = tmp["tsne1"]
    plt.scatter(x,y, color=cdict[label], label=label, s=3)
    
plt.legend()
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_hepato_type_s3.tiff",dpi=300)
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_hepato_type_s3.png",dpi=300)

plt.figure(figsize=(15,15))
clist = ["lime","brown","grey","blue","green","red"]

#['microT' 'SC' 'UC' 'PG' 'C' 'macroT']
print(test["histo"].unique())
cdict = dict(zip(test["histo"].unique(),clist))

for label in test["histo"].unique():

    tmp = test[test["histo"]==label].reset_index(drop=True)
        
    x = tmp["tsne0"]
    y = tmp["tsne1"]
    plt.scatter(x,y, color=cdict[label], label=label, s=3)
    
plt.legend()
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_histo_s3.tiff",dpi=300)
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_histo_s3.png",dpi=300)

plt.figure(figsize=(15,15))
clist = ["blue","red","lime","blue","green","grey"]

cdict = dict(zip(test["PGCC"].unique(),clist))

print(test["PGCC"].value_counts())
for label in test["PGCC"].unique():

    tmp = test[test["PGCC"]==label].reset_index(drop=True)
    
    color = cdict[label]
    if label==1:
        label="PGCC"
    else:
        label="without PGCC"
        
    x = tmp["tsne0"]
    y = tmp["tsne1"]
    plt.scatter(x,y, color=color, label=label, s=3)
    
plt.legend()
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_PGCC.tiff",dpi=300)
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_PGCC.png",dpi=300)




#"""
plt.figure(figsize=(18,15))
        
x = test["tsne0"]
y = test["tsne1"]
#plt.scatter(x,y, s=20)

cmap = plt.cm.get_cmap('coolwarm')

# 散布図をプロットし、predの値で色付け
sc = plt.scatter(x, y, c=test["pred__"], cmap=cmap, vmin=0, vmax=1,s=10)

# カラーバーを追加
cbar = plt.colorbar(sc,aspect=20)
#cbar.set_label('Pred Values')  # カラーバーのラベル
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_pred.tiff",dpi=300)
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_pred.png",dpi=300)

plt.figure(figsize=(18,15))

x = test["tsne0"]
y = test["tsne1"]
#plt.scatter(x,y, s=20)

cmap = plt.cm.get_cmap('coolwarm')

print(test["AFP"].min(),test["AFP"].max())
test["AFP_wo_log"]=10**test["AFP"]
min_ = test["AFP"].min()
max_ = test["AFP"].max()
median_ = test["AFP"].median()

test["AFPlog5"]=np.log(test["AFP_wo_log"]) / np.log(20)
print(min_,max_,median_)
import matplotlib.pyplot as plt


#plt.hist(test["AFP"],bins=100)
#plt.savefig(f"{DIR}/AFP_bin.png")
#0.2,5.7
# 散布図をプロットし、predの値で色付け
sc = plt.scatter(x, y, c=test["AFP"], cmap=cmap, vmin=min_-(max_-median_), vmax=max_,s=20)

#sc = plt.scatter(x, y, c=test["AFP"], cmap=cmap, vmin=min_, vmax=max_,s=20)

# カラーバーを追加
cbar = plt.colorbar(sc,aspect=20)
#cbar.set_label('Pred Values')  # カラーバーのラベル
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_AFP_m.tiff",dpi=300)
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_AFP_m.png",dpi=300)

plt.figure(figsize=(18,15))
sc = plt.scatter(x, y, c=test["AFPlog5"], cmap=cmap, vmin=test["AFPlog5"].min(), vmax=test["AFPlog5"].max(),s=20)

# カラーバーを追加
cbar = plt.colorbar(sc,aspect=20)
#cbar.set_label('Pred Values')  # カラーバーのラベル
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_AFP_5.tiff",dpi=300)
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_AFP_5.png",dpi=300)


test["AFPloge"]=np.log(test["AFP_wo_log"]) #/ np.log()

plt.figure(figsize=(18,15))
sc = plt.scatter(x, y, c=test["AFPloge"], cmap=cmap, vmin=test["AFPloge"].min()-(test["AFPloge"].max()-test["AFPloge"].median()), vmax=test["AFPloge"].max(),s=20)

# カラーバーを追加
cbar = plt.colorbar(sc,aspect=20)
#cbar.set_label('Pred Values')  # カラーバーのラベル
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_AFP_e.tiff",dpi=300)
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_AFP_e.png",dpi=300)



