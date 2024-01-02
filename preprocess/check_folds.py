"""
WSI_IDを確認

"""
import pandas as pd

from sklearn.metrics import roc_auc_score
df = pd.read_csv("/home/abebe9849/Nploid/src/outputs/2022-08-27/17-12-49/oof_iafoss000.csv")
df = pd.read_csv("/home/abebe9849/Nploid/src/outputs/2022-09-16/22-25-56/oof_nosiy_.csv")



th = 0.7

error_id = []
for id in df["WSI_ID"].unique():
    tmp = df[df["WSI_ID"]==id]
    #AUC_score = roc_auc_score(tmp["label"].values, tmp["pred"].values)
    #print(id,tmp["label"].values[0],tmp["pred"].values.mean())
    if abs(tmp["label"].values[0]-tmp["pred"].values.mean())>th:
        print(id,tmp["label"].values[0],tmp["pred"].values.mean())
        error_id.append(id)
        #print()


df_c = df[~df["WSI_ID"].isin(error_id)]
human_hard_id = ["AI007","AI016","AI027","AI032"]
df_human_hard = df[df["WSI_ID"].isin(human_hard_id)]
df_miss = df[df["WSI_ID"].isin(list(error_id))]
#df_miss["label"]=1-df_miss["label"].values

print(df_miss,sorted(error_id))
AUC_score = roc_auc_score(df_miss["label"].values, df_miss["pred"].values)
print(AUC_score)
AUC_score = roc_auc_score(df_human_hard["label"].values, df_human_hard["pred"].values)
print('["AI007","AI016","AI027","AI032"]',AUC_score)
df_collect =pd.concat([df_c,df_miss])

AUC_score = roc_auc_score(df_collect["label"].values, df_collect["pred"].values)

AUC_score = roc_auc_score(df_c["label"].values, df_c["pred"].values)
del df_collect["pred"]
print(AUC_score)
#df_collect.to_csv("/home/abebe9849/Nploid/src/oof_iafoss_noisy_collect_2.csv",index=False)

exit()
df = pd.read_csv("/home/abebe9849/Nploid/src/outputs/2022-09-10/17-39-49/oof_iafoss_seed1002.csv")
error_id2 = []
for id in df["WSI_ID"].unique():
    tmp = df[df["WSI_ID"]==id]
    #AUC_score = roc_auc_score(tmp["label"].values, tmp["pred"].values)
    #print(id,tmp["label"].values[0],tmp["pred"].values.mean())
    if abs(tmp["label"].values[0]-tmp["pred"].values.mean())>th:
        #print(id,tmp["label"].values[0],tmp["pred"].values.mean())
        error_id2.append(id)

df = pd.read_csv("/home/abebe9849/Nploid/src/outputs/2022-09-10/18-54-31/oof_iafoss_seed1000.csv")
error_id3 = []
for id in df["WSI_ID"].unique():
    tmp = df[df["WSI_ID"]==id]
    #AUC_score = roc_auc_score(tmp["label"].values, tmp["pred"].values)
    #print(id,tmp["label"].values[0],tmp["pred"].values.mean())
    if abs(tmp["label"].values[0]-tmp["pred"].values.mean())>th:
        #print(id,tmp["label"].values[0],tmp["pred"].values.mean())
        error_id3.append(id)


x = list(set(error_id)&set(error_id2)&set(error_id3))
#x = list(set(error_id)&set(error_id2))
x.sort()
print(x)
df = pd.read_csv("/home/abebe9849/Nploid/src/outputs/2022-08-27/17-12-49/oof_iafoss000.csv")
AUC_score = roc_auc_score(df["label"].values, df["pred"].values)

print(AUC_score)
import numpy as np
df_c = df[~df["WSI_ID"].isin(x)]
df_miss = df[df["WSI_ID"].isin(x)]
df_miss["label"]=1-df_miss["label"].values
AUC_score = roc_auc_score(df_miss["label"].values, df_miss["pred"].values)
#print(df_miss)
print("[‘AI017’, ‘AI023’, ‘AI053’] AUC",AUC_score)
AUC_score = roc_auc_score(df_c["label"].values, df_c["pred"].values)
print("other",AUC_score)

df_collect =pd.concat([df_c,df_miss])

AUC_score = roc_auc_score(df_collect["label"].values, df_collect["pred"].values)
print("collect",AUC_score)

del df_collect["pred"]
#df_collect.to_csv("/home/abebe9849/Nploid/src/nosiy_.csv",index=False)


