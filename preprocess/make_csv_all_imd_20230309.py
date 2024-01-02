import pandas as pd

import glob,cv2
import numpy as np
df = pd.read_csv("/home/abebe9849/Nploid/tabaitai.csv")
df_dict = dict(zip(df["WSI_ID"].values,df["WGD_by_FISH"].values))


all_ = glob.glob("/data/RSNA/Nploid/HE_midTiFF/*/*tif")
train_df = pd.DataFrame()
train_df["file_path"]=all_
train_df["WSI_ID"]=[x.split("/")[-2].split(" ")[0][:-1] for x in all_]


print(train_df["WSI_ID"].nunique())

select_img = glob.glob("/data/RSNA/Nploid/20230309/20230307_midtiff_additional_AI031-80/*/*tif")


train_df2 = pd.DataFrame()
train_df2["file_path"]=select_img
train_df2["WSI_ID"]=[x.split("/")[-2].split(" ")[0][:-1] for x in select_img]

print(train_df2["WSI_ID"].nunique())
print(train_df.shape)
train_df = pd.concat([train_df,train_df2]).reset_index(drop=True)
print(train_df["WSI_ID"].nunique())


print(train_df.shape)
train_df = train_df[~train_df["WSI_ID"].isin(["AI017","AI053","AI011"])].reset_index(drop=True)
print(train_df["WSI_ID"].unique(),train_df["WSI_ID"].nunique())


def func(x):
    return df_dict[x]
train_df["label"]=train_df["WSI_ID"].apply(func)

from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold

skf = StratifiedKFold(n_splits=5,random_state=2022,shuffle=True)
sgkf = StratifiedGroupKFold(n_splits=5,random_state=22,shuffle=True)

for fold, ( _, val_) in enumerate(sgkf.split(X=train_df, y=train_df.label,groups=train_df.WSI_ID)):
    train_df.loc[val_ , "fold"] = fold
    val_df = train_df[train_df["fold"]==fold]
train_df.to_csv("/home/abebe9849/Nploid/all_5folds20230309_v2.csv")

###
train_df["fold"]=-1

tmp= pd.read_csv("/home/abebe9849/Nploid/all_5folds.csv")

di = dict(zip(tmp["WSI_ID"],tmp["fold"]))
def func2(x):
    return di[x]
train_df["fold"]=train_df["WSI_ID"].apply(func2)


for i in range(5):
    val_df = train_df[train_df["fold"]==i]
    print(val_df["label"].value_counts()) 
    print(sorted(val_df["WSI_ID"].unique()))


#train_df.to_csv("/home/abebe9849/Nploid/all_5folds20230309.csv")