from lifelines import datasets
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve
from lifelines.plotting import add_at_risk_counts

from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize
import glob
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import plot_lifetimes
from numpy.random import uniform, exponential
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression

DIR = "/home/abebe9849/Nploid/src/outputs/2023-03-13/resnet50d_all"



DIR_s = ["/home/abebe9849/Nploid/src/outputs/2023-03-10/b0_aug_all",
         "/home/abebe9849/Nploid/src/outputs/2023-03-10/HIPT_unfreeze1_aug_all",
         "/home/abebe9849/Nploid/src/outputs/2023-03-10/HIPT_unfreeze2_aug_all",
         "/home/abebe9849/Nploid/src/outputs/2023-03-12/b0_aug_gray_select",
         "/home/abebe9849/Nploid/src/outputs/2023-03-12/b0_aug_select",
         "/home/abebe9849/Nploid/src/outputs/2023-03-12/b3_aug_select",
         "/home/abebe9849/Nploid/src/outputs/2023-03-12/HIPT_unfreeze1_aug_select",
         "/home/abebe9849/Nploid/src/outputs/2023-03-12/HIPT_unfreeze2_aug_select",
         "/home/abebe9849/Nploid/src/outputs/2023-03-13/b0_aug_gray_all",
         "/home/abebe9849/Nploid/src/outputs/2023-03-13/densenet121_all",
         "/home/abebe9849/Nploid/src/outputs/2023-03-13/resnet50d_all",
         "/home/abebe9849/Nploid/src/outputs/2023-02-28/densenet121_select",
         "/home/abebe9849/Nploid/src/outputs/2023-02-28/resnet50d_select",
         ]


def test_func(df,test,oof,fold):
    emb_s = []
    for patient in test["WSI_ID"].unique():
        tmp =test[test["WSI_ID"]==patient]["pred"]
        tmp = tmp.mean()
        #tmp = np.percentile(tmp.to_numpy(), 95)
        emb_s.append(tmp)
        

        
    dict_= dict(zip(test["WSI_ID"].unique(),emb_s))



    def func(x):
        out = dict_[x]
        return out

    df["pred"]=df["AI No."].apply(func)



    df = df[df["outcome_bi"].isin([0,1])].reset_index(drop=True)
    print(df.shape)



    for i in range(1,5):
        t = i*365
        gt = np.where(df["OS"]>=t,1,0)
        if  len(np.unique(gt))==1:continue
        AUC_score = roc_auc_score(gt, df["pred"].to_numpy())
        print(i,AUC_score)




    true_y = oof["label"].to_numpy()
    pred_y = oof["pred"].to_numpy()



    def func_y_index(x):

        # Calculate the confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_y, pred_y >= x).ravel()

        # Calculate the sensitivity and specificity
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        # Calculate the Youden index
        j_index = sensitivity + specificity - 1
        return -j_index
    result = minimize(func_y_index, x0=np.array([0.5]), method='Nelder-Mead')
    print(result['x'].item(),func_y_index(result['x'].item())*-1)


    Y_index = result['x'].item()
    fpr, tpr, thresholds = roc_curve(true_y, pred_y)
    distances = np.sqrt(fpr**2 + (tpr-1)**2)
    min_index = np.argmin(distances)
    optimal_threshold = thresholds[min_index]
    print(optimal_threshold)
    
    def plot_KM(type,TH):
        high = df[df["pred"]>=TH]
        low = df[df["pred"]<TH]
        Npllid_rate = np.round(high.shape[0]/df.shape[0],3)
        ax = plt.subplot(111)
        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()
        try:
            kmf_high.fit(durations=high.OS, event_observed=high.outcome_bi,label="high prrd")
            ax = kmf_high.plot_survival_function(ax=ax,ci_alpha=0.05)#CIつき
        except Exception as e:
            print(e)
            plt.close()
            return 
        
        #kmf.survival_function_.plot(ax=ax)
        
        try:
            kmf_low.fit(durations=low.OS, event_observed=low.outcome_bi,label="low_pred")
            ax = kmf_low.plot_survival_function(ax=ax,ci_alpha=0.05)
        except Exception as e:
            print(e)
            plt.close()
            return 
        
        #kmf.survival_function_.plot(ax=ax)
        ax.set_ylim(0,1)
        ax.set_xticks(range(0,2200,200))
        add_at_risk_counts(kmf_high, kmf_low, ax=ax)
        plt.tight_layout()


        result = logrank_test(high.OS, low.OS, high.outcome_bi, low.outcome_bi)
        result.print_summary()
        print(result.p_value)
        plt.title(f" th {np.round(TH,3)} p_value{result.p_value} high/all{Npllid_rate}")
        DIR_ = DIR.split("/")[-1]
        plt.savefig(f"{DIR}/KM_type{type}_th{TH}_{DIR_}.png")
        plt.close()
        return 


    #plot_KM(f"{fold}_05",0.5)
    plot_KM(f"{fold}_35%",df["pred"].quantile(q=0.65))
    plot_KM(f"{fold}_Youden",Y_index)
    plot_KM(f"{fold}_distance",optimal_threshold)
    return 



def cross_validation_with_platt_scaling(oof_df,test_df,df_tmp):
    
    oof_df = oof_df.rename(columns={'pred': 'pred_before_platt_scaling'})
    test_df = test_df.rename(columns={'pred': 'pred_before_platt_scaling'})
    
    new_oof = []
    for fold in range(5):
        print(fold,"FOLD")
        tra_df = oof_df[oof_df["fold"]!=fold].reset_index(drop=True)
        val_df = oof_df[oof_df["fold"]==fold].reset_index(drop=True)


        # ロジスティック回帰モデルの学習
        platt_clf = LogisticRegression()
        y_pred = val_df["pred_before_platt_scaling"].to_numpy()
        platt_clf.fit(y_pred.reshape(-1, 1), val_df["label"].to_numpy())

        # 予測結果の確率値の再キャリブレーション
        val_df["pred"] = platt_clf.predict_proba(y_pred.reshape(-1, 1))[:, 1]
        new_oof.append(val_df)
        
        test_df["pred"] = platt_clf.predict_proba(test_df["pred_before_platt_scaling"].to_numpy().reshape(-1, 1))[:, 1]
        
        
        test_func(df_tmp,test_df,val_df,fold)
        
        

    new_oof = pd.concat(new_oof).reset_index(drop=True)
    return new_oof


DIR_s = [
         "/home/abebe9849/Nploid/src/outputs/2023-06-06/HIPT_unfreeze2",

         ]
for DIR in DIR_s:
    try:
        test_df =pd.read_csv(f"{DIR}/test.csv")
        path = glob.glob(f"{DIR}/*oof*all*csv")[0]
        print(path)
        oof = pd.read_csv(path)
    except Exception as e:
        print(e)
        test_df =pd.read_csv(f"{DIR}/test_wo_duplicate.csv")
        oof = pd.read_csv(glob.glob(f"{DIR}/*oof*csv")[0])
    df_clinical = pd.read_csv("clinical_data.csv")

    test170 = pd.read_csv("test_clinical_data.csv")
    def func(x):
        return x.replace("HL","AI")
    test170["AI No."] = test170["Pt No."].apply(func)
    df_clinical = df_clinical[df_clinical["AI No."].isin(test_df["WSI_ID"].unique())]
    df_clinical = df_clinical[df_clinical["AI No."].isin(test170["AI No."].unique())]
    test_func(df_clinical,test_df,oof,"original")
