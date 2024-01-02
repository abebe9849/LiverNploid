"""
20221216に届いたデータについて


"""
import  os,time,glob,cv2
import collections
import pandas as pd
import numpy as np

folds = pd.read_csv("/home/abebe9849/Nploid/all_5folds.csv")
folds_uni =folds["WSI_ID"].unique()
print(folds.columns)

df = pd.DataFrame()
df["file_path"]=glob.glob("/data/RSNA/Nploid_test/20221216多倍体データ/20221120_MidTIFF/*/*")
path_  = df["file_path"].values[0]
def func(x):
    x = x.split("/")[-2].split("_")[2]
    return "AI"+x
df["WSI_ID"]= np.vectorize(func)(df["file_path"])

ex_uni = df["WSI_ID"].unique()
all_ = pd.concat([df,folds]).reset_index(drop=True)
print(df["WSI_ID"].unique())
print(all_["WSI_ID"].nunique())

print(set(ex_uni)&set(folds_uni))




