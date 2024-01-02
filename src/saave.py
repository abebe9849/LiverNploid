import pandas as pd
import sys,os,time,glob


def func(x):
    path_ = x.split("/")[-1]
    os.makedirs("/home/abebe9849/Nploid/src/SAVE4/"+path_,exist_ok=True)
    os.chdir("/home/abebe9849/Nploid/src/SAVE4/"+path_)
    
    folds_all = pd.read_csv(x+"/oof_all.csv")
    
    assert folds_all["WSI_ID"].nunique()==44,folds_all["WSI_ID"].nunique()
    del folds_all["fold"]
    del folds_all["select"]
    folds_all.to_csv("oof_all.csv",index=False)
    
    test = pd.read_csv(x+"/test.csv")
    test = test[test["WSI_ID"]!="AI053"].reset_index(drop=True) #メールにて除くようにとあったので
    
    
    test.to_csv("test.csv",index=False)
    
    test_g = test.groupby("WSI_ID")["pred"].describe().reset_index()
    test_g.to_csv("test_groupby.csv",index=False)

    

    
    
    
    

    
    
    
    
    
    return 

    
    
    
func("/home/abebe9849/Nploid/src/outputs/2023-03-12/b0_aug_select")
func("/home/abebe9849/Nploid/src/outputs/2023-03-12/b0_aug_gray_select")

func("/home/abebe9849/Nploid/src/outputs/2023-03-12/HIPT_unfreeze1_aug_select")
func("/home/abebe9849/Nploid/src/outputs/2023-03-12/HIPT_unfreeze2_aug_select")