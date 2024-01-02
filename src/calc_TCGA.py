import polars as pl
import pandas as pd
import numpy as np
import glob,time
from sklearn.metrics import roc_auc_score

DIR = "/home/abebe9849/Nploid/src/outputs/2023-03-12/HIPT_unfreeze3_aug_select/TCGA"
"""
HIPT_unfreeze1_aug_select

0.6424465733235077
0.75 :0.6585298452468681
0.65 0.6569086219602063
0.7 0.6587877671333824
0.6 0.6545136330140015
0.8 0.6591009579955784
0.825 0.6576271186440678
"""


start = time.time()
df_GT = pl.read_csv("/home/abebe9849/Nploid/preprocess/TCGA_GT.csv")

file_ids = []
mean_preds = []
for csv_ in glob.glob(f"{DIR}/*csv"):
    file_id = csv_.split("/")[-1].split(".csv")[0]
    tmp = pl.read_csv(csv_)
    mean_pred = tmp["pred"].mean()
    mean_pred = tmp["pred"].quantile(0.79)
    mean_preds.append(mean_pred)
    file_ids.append(file_id)
    
    
tmp_df = pl.DataFrame({"FILE_ID": file_ids, 
              "pred":mean_preds})

dict_ = dict(zip(file_ids,mean_preds))


def func(x):
    try:
        out = dict_[x]
        return out
    except:
        return -1
    
df_GT = df_GT.join(tmp_df,on="FILE_ID")

print(df_GT.shape)
print(roc_auc_score(df_GT["多倍体癌"].to_numpy(),df_GT["pred"].to_numpy()))
#0.5546425939572586 contrast_limit075
#0.49602063375092115 b0_aug_select
#0.6424465733235077 HIPT_unfreeze1_aug_select
#0.5337509211495948 b5
print(time.time()-start)
