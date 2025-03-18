import hydra
from omegaconf import DictConfig, OmegaConf
import sys,gc,os,random,time,math,glob
import matplotlib.pyplot as plt
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict, Counter
from  torch.cuda.amp import autocast, GradScaler 
import cv2,pyvips,timm
from sklearn.metrics import roc_auc_score
from PIL import Image
import numpy as np
import pandas as pd
import scipy as sp
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold,GroupKFold
from sklearn.metrics import log_loss
from functools import partial
from tqdm import tqdm
from sklearn.metrics import precision_score,recall_score,f1_score,log_loss
from  sklearn.metrics import accuracy_score as acc
import torch
import torch.nn as nn
from torch.optim import Adam, SGD,AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau,CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip,RandomGamma, RandomRotate90,GaussNoise,Cutout,RandomBrightnessContrast,RandomContrast,Resize
from albumentations.pytorch import ToTensorV2
import transformers as T

import albumentations as A
import vision_transformer as vits


### my utils
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d


# From: https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/pooling.py
def gem_1d(x, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1),)).pow(1./p)


def gem_2d(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


def gem_3d(x, p=3, eps=1e-6):
    return F.avg_pool3d(x.clamp(min=eps).pow(p), (x.size(-3), x.size(-2), x.size(-1))).pow(1./p)


_GEM_FN = {
    1: gem_1d, 2: gem_2d, 3: gem_3d
}


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6, dim=2):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        return _GEM_FN[self.dim](x, p=self.p, eps=self.eps)

class AdaptiveConcatPool1d(nn.Module):

    def forward(self, x):
        return torch.cat((F.adaptive_avg_pool1d(x, 1), F.adaptive_max_pool1d(x, 1)), dim=1)


class AdaptiveConcatPool2d(nn.Module):

    def forward(self, x):
        return torch.cat((F.adaptive_avg_pool2d(x, 1), F.adaptive_max_pool2d(x, 1)), dim=1)


class AdaptiveConcatPool3d(nn.Module):

    def forward(self, x):
        return torch.cat((F.adaptive_avg_pool3d(x, 1), F.adaptive_max_pool3d(x, 1)), dim=1)

###

import logging
#from mylib.
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


from timm.data.transforms import RandomResizedCropAndInterpolation
#### dataset ==============
class TrainDataset(Dataset):
    def __init__(self, df,CFG,train=True,transform1=None):
        self.df = df
        self.transform = transform1
        self.CFG = CFG
        self.train = train
        self.image_ids =self.df['file_path'].to_numpy()
        self.labels =self.df['label'].to_numpy()
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        imgs = glob.glob(f"{image_id}*tif")
        imgs =  np.stack([self.transform(image=cv2.imread(img)[:,:,::-1])['image']  for img in imgs])
        image = torch.from_numpy(imgs.transpose(0,3,1,2)).float()

        label = self.labels[idx]

        
        return image, torch.tensor(label).float()
import skimage.util

class TrainDataset(Dataset):
    def __init__(self, df,CFG,train=True,transform1=None):
        self.df = df
        self.transform = transform1
        self.CFG = CFG
        self.train = train
        self.labels = self.df['label'].to_numpy()
        self.image_paths = self.df["file_path"].values
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        imgs =  cv2.imread(self.image_paths[idx])[:,:,::-1]#2048,2048,3
        imgs = skimage.util.view_as_blocks(imgs, (256, 256, 3)).squeeze()
        imgs = np.concatenate(imgs,0)#64,256,256,3
        image = torch.from_numpy(imgs.transpose(0,3,1,2)).float()

        label = self.labels[idx]

        
        return image, torch.tensor(label).float()

class TestDataset(Dataset):
    def __init__(self, df,CFG,train=True,transform1=None):
        self.df = df
        self.transform = transform1
        self.CFG = CFG
        self.image_paths = self.df["file_path"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        imgs =  cv2.imread(self.image_paths[idx])[:,:,::-1]#2048,2048,3
        imgs = skimage.util.view_as_blocks(imgs, (256, 256, 3)).squeeze()
        imgs = np.concatenate(imgs,0)#64,256,256,3
        imgs =  np.stack([self.transform(image=img)['image']  for img in imgs])
        image = torch.from_numpy(imgs.transpose(0,3,1,2)).float()

        return image


#### dataset ==============

#### augmentation ==============


def get_transforms(*, data,CFG):
    if data == 'train':
        return Compose([
            A.HorizontalFlip(p=CFG.aug.HorizontalFlip.p),
            A.VerticalFlip(p=CFG.aug.VerticalFlip.p),
            A.RandomRotate90(p=CFG.aug.RandomRotate90.p),
            A.ShiftScaleRotate(
                shift_limit=CFG.aug.ShiftScaleRotate.shift_limit,
                scale_limit=CFG.aug.ShiftScaleRotate.scale_limit,
                rotate_limit=CFG.aug.ShiftScaleRotate.rotate_limit,
                p=CFG.aug.ShiftScaleRotate.p),
            A.RandomBrightnessContrast(
                brightness_limit=CFG.aug.RandomBrightnessContrast.brightness_limit,
                contrast_limit=CFG.aug.RandomBrightnessContrast.contrast_limit,
                p=CFG.aug.RandomBrightnessContrast.p),
            A.CLAHE(
                clip_limit=(1,4),
                p=CFG.aug.CLAHE.p),
            A.OneOf([
                A.JpegCompression(),
                A.Downscale(scale_min=0.1, scale_max=0.15),
                ], p=CFG.aug.compress.p),
            A.CoarseDropout(max_holes=CFG.aug.CoarseDropout.max_holes, max_height=CFG.aug.CoarseDropout.max_height, max_width=CFG.aug.CoarseDropout.max_width, p=CFG.aug.CoarseDropout.p),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
    elif data == 'valid':
        return Compose([
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])


#### augmentation ==============

#### model ================
SEQ_POOLING = {
    'gem': GeM(dim=2),
    'concat': AdaptiveConcatPool2d(),
    'avg': nn.AdaptiveAvgPool2d(1),
    'max': nn.AdaptiveMaxPool2d(1)
}

class Model_HIPT(nn.Module):
    def __init__(self, base_model='tf_efficientnet_b0_ns',freeze=1):
        super(Model_HIPT, self).__init__()
        self.base_model = "dino_vit_s"
        
        
        checkpoint_key = "teacher"
        pretrained_weights = "/home/abebe9849/Nploid/dino/vit256_small_dino.pth"
        self.model = vits.__dict__["vit_small"](patch_size=16, num_classes=0)
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        print(state_dict.keys())
        if checkpoint_key is not None and checkpoint_key in state_dict:
            state_dict = state_dict[checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)
        
        for _, p in self.model.named_parameters():
            p.requires_grad = False
        for _, p in self.model.head.named_parameters():
            p.requires_grad = True

        unfreeze = freeze
        for n, p in self.model.blocks.named_parameters():
            if int(n.split(".")[0])>=(12-unfreeze):
                p.requires_grad = True
        
        avgpool_patchtokens = 0
        n_last_blocks = 2
        self.n_last_blocks = n_last_blocks
        nc = self.model.embed_dim * (n_last_blocks + int(avgpool_patchtokens))
        self.gru = nn.GRU(nc, 512, bidirectional=True, batch_first=True, num_layers=2)

        nc*=64
        self.head = nn.Sequential(nn.Linear(nc,512),
                        nn.ReLU(), nn.Dropout(0.5),nn.Linear(512,1))
        self.exam_predictor = nn.Linear(512*2, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        


        
    def forward(self, input1):
        shape = input1.size()
        batch_size = shape[0]
        n = shape[1]
        input1 = input1.view(-1,shape[2],shape[3],shape[4])

        intermediate_output = self.model.get_intermediate_layers(input1, self.n_last_blocks)
        x = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        embeds, _ = self.gru(x.view(batch_size,n,x.shape[1]))
        embeds = self.pool(embeds.permute(0,2,1))[:,:,0]
        y = self.exam_predictor(embeds)
        
        return y
class Model_HIPT(nn.Module):
    def __init__(self, base_model='tf_efficientnet_b0_ns',freeze=1):
        super(Model_HIPT, self).__init__()
        self.base_model = "dino_vit_s"
        
        
        checkpoint_key = "teacher"
        pretrained_weights = "/home/abebe9849/Nploid/dino/vit256_small_dino.pth"
        #pretrained_weights = "/home/abebe9849/Nploid/dino/exp000/checkpoint.pth"
        self.model = vits.__dict__["vit_small"](patch_size=16, num_classes=0)
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        print(state_dict.keys())
        if checkpoint_key is not None and checkpoint_key in state_dict:
            state_dict = state_dict[checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)
        
        for _, p in self.model.named_parameters():
            p.requires_grad = False
        for _, p in self.model.head.named_parameters():
            p.requires_grad = True

        unfreeze = freeze
        for n, p in self.model.blocks.named_parameters():
            if int(n.split(".")[0])>=(12-unfreeze):
                p.requires_grad = True
        
        avgpool_patchtokens = 0
        n_last_blocks = 1
        self.n_last_blocks = n_last_blocks
        nc = self.model.embed_dim * (n_last_blocks + int(avgpool_patchtokens))
        self.gru = nn.GRU(nc, 512, bidirectional=True, batch_first=True, num_layers=2)

        nc*=64
        self.head = nn.Sequential(nn.Linear(nc,512),
                        nn.ReLU(), nn.Dropout(0.5),nn.Linear(512,1))
        self.exam_predictor = nn.Linear(512*2, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        


        
    def forward(self, input1):
        shape = input1.size()
        batch_size = shape[0]
        n = shape[1]
        input1 = input1.view(-1,shape[2],shape[3],shape[4])

        intermediate_output = self.model.get_intermediate_layers(input1, self.n_last_blocks)
        x = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        embeds, _ = self.gru(x.view(batch_size,n,x.shape[1]))
        embeds = self.pool(embeds.permute(0,2,1))[:,:,0]
        y = self.exam_predictor(embeds)
        #x = x.view(batch_size,x.shape[1]*n)
        #y = self.head(x)
        
        return y
    
    
    def get_emb(self, input1):
        shape = input1.size()
        batch_size = shape[0]
        n = shape[1]
        input1 = input1.view(-1,shape[2],shape[3],shape[4])

        intermediate_output = self.model.get_intermediate_layers(input1, self.n_last_blocks)
        x = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        embeds, _ = self.gru(x.view(batch_size,n,x.shape[1]))
        embeds = self.pool(embeds.permute(0,2,1))[:,:,0]
        y = self.exam_predictor(embeds)        
        return y,embeds



def inf_func(models, valid_loader, device):
    #models = [m for m in models]
    for model in models:
        model.eval()

    preds = []
    
    embs = [[],[],[],[],[]]

    tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

    for i, images in tk1:
        if i==10:s = time.time()
        elif i==20:print(time.time()-s)
        images = images.to(device,non_blocking=True)
        #print(images.shape)
        with torch.no_grad():
            with autocast():
                y_preds = []
                for idx,m in enumerate(models):
                    pred,emb  = m.get_emb(images.float())
                    y_preds.append(pred.sigmoid())
                    embs[idx].append(emb)
                
                #print(y_preds)
                y_preds  = torch.stack(y_preds).mean(0)

        preds.append(y_preds)
        
    for idx  in range(5):
        embs[idx] = torch.cat(embs[idx]).to('cpu').numpy()
    preds = torch.cat(preds).to('cpu').numpy()

    return preds,embs

def val_func(model, valid_loader, device):
    model.eval()

    preds = []

    tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

    for i, images in tk1:
        images = images.to(device,non_blocking=True)
        #print(images.shape)
        with torch.no_grad():
            with autocast():
                y_preds = model(images.float()).sigmoid()

        preds.append(y_preds)
    preds = torch.cat(preds).to('cpu').numpy()

    return preds

def submit(CFG,test,DIR):
    torch.cuda.set_device(CFG.general.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = []
    for fold in   range(5):
        model = Model_HIPT(base_model="dino_vit_s",freeze=1).to(device)
        model.load_state_dict(torch.load(f"{DIR}/fold{fold}_{CFG.general.exp_num}__dino_best_AUC.pth", map_location="cpu"),strict=False)

        models.append(model)
        
    valid_dataset = TestDataset(test,train=False,
                                 transform1=get_transforms(data='valid',CFG=CFG),CFG=CFG)# 
    valid_loader = DataLoader(valid_dataset, batch_size=24, shuffle=False, num_workers=12,pin_memory=True)
    tets_preds,embs = inf_func(models, valid_loader, device)
    
    test["pred"]=tets_preds
    
    
    
    return test,embs
    
    
        
        
def submit_folds(CFG,folds,DIR):
    torch.cuda.set_device(CFG.general.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    folds_all = pd.read_csv("/home/abebe9849/Nploid/preprocess/select_all0401.csv")

    oof = pd.DataFrame()
    for fold in   range(5):
        
        val_ID = folds["WSI_ID"].unique()
        model = Model_HIPT(base_model="dino_vit_s",freeze=1).to(device)
        model.load_state_dict(torch.load(f"{DIR}/fold{fold}_{CFG.general.exp_num}__HIPT_best_AUC.pth", map_location="cpu"),strict=False)
        
        val_folds = folds_all[folds_all["WSI_ID"].isin(val_ID)].reset_index(drop=True)
        valid_dataset = TestDataset(val_folds,train=False,
                                 transform1=get_transforms(data='valid',CFG=CFG),CFG=CFG)# 
        valid_loader = DataLoader(valid_dataset, batch_size=24, shuffle=False, num_workers=12,pin_memory=True)
        test_preds = val_func(model, valid_loader, device)
    
        val_folds["pred"]=test_preds
        oof = pd.concat([oof,val_folds])
        
    oof.to_csv(f"{DIR}/oof_all.csv",index=False)
    
    return oof

    



df = pd.DataFrame()
df["file_path"]=glob.glob("/data/RSNA/Nploid_test/20221216多倍体データ/20221120_MidTIFF/*/*")

path_  = df["file_path"].values[0]
def func(x):
    x = x.split("/")[-2].split("_")[2]
    return "AI"+x
df["WSI_ID"]= np.vectorize(func)(df["file_path"])

DIR = "/home/abebe9849/Nploid/src/outputs/2023-03-12/dino_unfreeze1_aug_select" 


log = logging.getLogger(__name__)
@hydra.main(config_path=f"{DIR}/.hydra/",config_name="config")
def main(CFG : DictConfig) -> None:

    seed_torch(seed=CFG.general.seed)
    CFG.general.device = 1
    #os.environ["CUDA_VISIBLE_DEVICES"]=f"{CFG.general.device}"
    log.info(f"===============exp_num{CFG.general.exp_num}============")

    log.info(f"{DIR}")
    
    df_tmp = pd.DataFrame()
    df_tmp["file_path"]=glob.glob("/data/RSNA/Nploid_test/20221216多倍体データ/20221120_MidTIFF/*/*")

    path_  = df_tmp["file_path"].values[0]
    def func(x):
        x = x.split("/")[-2].split("_")[2]
        return "AI"+x
    df_tmp["WSI_ID"]= np.vectorize(func)(df_tmp["file_path"])



    df_tmp = df_tmp[~df_tmp["WSI_ID"].isin(folds["WSI_ID"].unique())].reset_index(drop=True)
    gt = pd.read_csv("/home/abebe9849/Nploid/20230224_WGDbyFISH_pred_final.csv")
    def func(x):
        return "AI"+x[2:]
    gt["WSI_ID"]= gt["Pt_No"].apply(func)
    df_tmp = df_tmp[df_tmp["WSI_ID"].isin(gt["WSI_ID"].unique())].reset_index(drop=True)
    
    
    test_df,embs = submit(CFG,df_tmp,DIR)

    test_df.to_csv(f"{DIR}/test.csv",index=False)

    
    tset_g = test_df.groupby("WSI_ID")["pred"].describe().reset_index()
    tset_g.to_csv(f"{DIR}/test_groupby.csv",index=False)


if __name__ == "__main__":
    main()
