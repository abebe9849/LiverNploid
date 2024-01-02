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


### my utils
from code_factory.pooling import GeM,AdaptiveConcatPool2d
from code_factory.augmix import RandomAugMix
from code_factory.gridmask import GridMask
from code_factory.fmix import *
from code_factory.loss_func import *

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


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['file_path'].values[idx]
        image = cv2.imread(file_path)[:,:,::-1]

        image = self.transform(image=image)['image']
        image = torch.from_numpy(image.transpose(2,0,1)).float()
        label_ = self.df["label"].values[idx]
        label = torch.tensor(label_).float()
        
        return image, label


class TestDataset(Dataset):
    def __init__(self, df,CFG,train=True,transform1=None):
        self.df = df
        self.transform = transform1
        self.CFG = CFG

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['file_path'].values[idx]
        image = cv2.imread(file_path)

        #image = cv2.resize(image, (512, 512))

        image = self.transform(image=image)['image']
        image = torch.from_numpy(image.transpose(2,0,1)).float()
        return image


#### dataset ==============

#### augmentation ==============


def get_transforms(*, data,CFG):
    if data == 'train':
        return Compose([
            #Resize(CFG.preprocess.size,CFG.preprocess.size),
            #A.augmentations.crops.transforms.CenterCrop(CFG.preprocess.size*0.9,CFG.preprocess.size*0.9),
            #A.crops.transforms.RandomResizedCrop(CFG.preprocess.size,CFG.preprocess.size),
            #A.crops.transforms.RandomCrop(CFG.preprocess.size,CFG.preprocess.size),
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
                A.ImageCompression(),
                A.Downscale(scale_min=0.1, scale_max=0.15),
                ], p=CFG.aug.compress.p),
            GridMask(
                num_grid=CFG.aug.GridMask.num_grid,p=CFG.aug.GridMask.p),
            A.CoarseDropout(max_holes=CFG.aug.CoarseDropout.max_holes, max_height=CFG.aug.CoarseDropout.max_height, max_width=CFG.aug.CoarseDropout.max_width, p=CFG.aug.CoarseDropout.p),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
    elif data == 'valid':
        return Compose([
            #Resize(CFG.preprocess.size,CFG.preprocess.size),
            #A.augmentations.crops.transforms.CenterCrop(int(CFG.preprocess.size*0.9),int(CFG.preprocess.size*0.9)),
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

class Model(nn.Module):
    def __init__(self,CFG, num_classes=1, base_model='tf_efficientnet_b0_ns',pool="avg",pretrain=True):
        super(Model, self).__init__()
        self.base_model = base_model #"str"
        self.CFG = CFG
        self.model = timm.create_model(self.CFG.model.name, pretrained=True, num_classes=1
        ,drop_path_rate=CFG.model.drop_path_rate
        ,drop_rate=CFG.model.drop_rate
        )
        if CFG.model.stride==1 and "swin" not in self.base_model:
            self.model.conv_stem = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1, bias=False)


        nc = self.model.num_features
        if pool in ('avg','concat','gem','max'):
            self.avgpool = SEQ_POOLING[pool]
            if pool == "concat":
                nc *= 2
        self.last_linear = nn.Linear(nc,num_classes)
        #self.last_linear = nn.Sequential(nn.Linear(nc,512),nn.ReLU(),nn.BatchNorm1d(512),nn.Linear(512,num_classes))#,nn.Dropout(0.5)
    def forward(self, input1):
        x = self.model.forward_features(input1)
        if "vit" in self.CFG.model.name or "swin" in self.CFG.model.name or "deit" in self.CFG.model.name:
            feature = x
        else:
            feature = self.avgpool(x).view(input1.size()[0], -1)
        y = self.last_linear(feature)
        return y

#model = Model_iafoss("dino_vit_s")


#### model ================


def train_fn(CFG,fold,folds,test_pl=0):

    #nvnn_transform = A.load("/home/u094724e/NIH/siim2021_nvnn/pipeline1/configs/aug/s_0220/0220_hf_cut_sm2_0.75_384_v1.yaml", data_format='yaml')

    torch.cuda.set_device(CFG.general.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"### fold: {fold} ###")
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index
    val_folds = folds.loc[val_idx].reset_index(drop=True)
    tra_folds = folds.loc[trn_idx]
    if type(test_pl)!=type(0):
        tra_folds = pd.concat([tra_folds,test_pl]).reset_index(drop=True)
    train_dataset = TrainDataset(tra_folds,train=True, 
                                 transform1=get_transforms(data='train',CFG=CFG),CFG=CFG)#get_transforms(data='train',CFG=CFG)
    valid_dataset = TrainDataset(val_folds,train=False,
                                 transform1=get_transforms(data='valid',CFG=CFG),CFG=CFG)#



    train_loader = DataLoader(train_dataset, batch_size=CFG.train.batch_size, shuffle=True, num_workers=8,pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.train.batch_size, shuffle=False, num_workers=8,pin_memory=True)

    ###  model select ============
    model = Model(CFG).to(device)
    # ============


    ###  optim select ============
    if CFG.train.optim=="adam":
        optimizer = Adam(model.parameters(), lr=CFG.train.lr, amsgrad=False)
    elif CFG.train.optim=="adamw":
        optimizer = AdamW(model.parameters(), lr=CFG.train.lr,weight_decay=5e-5)
    # ============

    ###  scheduler select ============
    if CFG.train.scheduler.name=="cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG.train.epochs, eta_min=CFG.train.scheduler.min_lr)
    elif CFG.train.scheduler.name=="cosine_warmup":
        scheduler =T.get_cosine_schedule_with_warmup(optimizer,
        num_warmup_steps=len(train_loader)*CFG.train.scheduler.warmup,
        num_training_steps=len(train_loader)*CFG.train.epochs)

    # ============

    ###  loss select ============
    criterion=nn.BCEWithLogitsLoss()
    print(criterion)
    ###  loss select ============

    sigmoid = nn.Sigmoid()
    scaler = torch.cuda.amp.GradScaler()
    best_score = 0
    best_loss = np.inf
    best_preds = None
        
    for epoch in range(CFG.train.epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.

        tk0 = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (images, labels) in tk0:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)

            ### mix系のaugumentation=========
            rand = np.random.rand()
            ##mixupを終盤のepochでとめる
            if epoch+1 >=CFG.train.without_hesitate:
                rand=1

            if CFG.augmentation.mix_p>rand and CFG.augmentation.do_mixup:
                images, y_a, y_b, lam = mixup_data(images, labels,alpha=CFG.augmentation.mix_alpha)
            elif CFG.augmentation.mix_p>rand and CFG.augmentation.do_cutmix:
                images, y_a, y_b, lam = cutmix_data(images, labels,alpha=CFG.augmentation.mix_alpha)
            elif CFG.augmentation.mix_p>rand and CFG.augmentation.do_resizemix:
                images, y_a, y_b, lam = resizemix_data(images, labels,alpha=CFG.augmentation.mix_alpha)
            elif CFG.augmentation.mix_p>rand and CFG.augmentation.do_fmix:
                images, y_a, y_b, lam = fmix_data(images, labels,alpha=CFG.augmentation.mix_alpha)
            ### mix系のaugumentation おわり=========

            if CFG.train.amp:
                with autocast():
                    y_preds = model(images)
                    if CFG.augmentation.mix_p>rand:
                        loss_ = mixup_criterion(criterion, y_preds, y_a.view(-1,1), y_b.view(-1,1), lam)
                    else:
                        loss_ = criterion(y_preds,labels.view(-1,1))

                    loss=loss_

                scaler.scale(loss).backward()

                if (i+1)%CFG.train.ga_accum==0 or i==-1:
                    scaler.step(optimizer)
                    scaler.update()

                        
                    if CFG.train.scheduler.name=="cosine_warmup":
                        scheduler.step()

       

            if CFG.train.scheduler.name=="cosine":
                scheduler.step()


            avg_loss += loss.item() / len(train_loader)
        model.eval()
        avg_val_loss = 0.
        LOGITS = []
        valid_labels = []
        tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

        for i, (images, labels) in tk1:
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                
                logits = model(images)
                loss =  criterion(logits,labels.view(-1,1))

            valid_labels.append(labels)
            LOGITS.append(logits.detach())
            avg_val_loss += loss.item() / len(valid_loader)
        preds = torch.sigmoid(torch.cat(LOGITS)).cpu().numpy().squeeze() 
        valid_labels = torch.cat(valid_labels).cpu().numpy()

        print(preds.shape,valid_labels.shape)
        print(valid_labels.mean(axis=0))

        AUC_score = roc_auc_score(valid_labels, preds)


        elapsed = time.time() - start_time

        log.info(f"AUC_score  {AUC_score}")
        log.info(f'  Epoch {epoch+1} - avg_train_loss: {avg_loss:.6f}  avg_val_loss: {avg_val_loss:.6f}  time: {elapsed:.0f}s')

        if best_loss>avg_val_loss:#pr_auc best
            best_loss = avg_val_loss
            log.info(f'  Epoch {epoch+1} - Save Best loss: {best_loss:.4f}')
            torch.save(model.state_dict(), f'fold{fold}_{CFG.general.exp_num}_best_loss.pth')
        if AUC_score>best_score:#pr_auc best
            best_score = AUC_score
            log.info(f'  Epoch {epoch+1} - Save Best AUC: {best_score:.4f}')
            best_preds = preds
            torch.save(model.state_dict(), f'fold{fold}_{CFG.general.exp_num}_best_AUC.pth')

        val_folds["pred"]=best_preds

    return best_preds, valid_labels,val_folds



def eval_func(model, valid_loader, device,CFG):
    model.to(device) 
    model.eval()

    valid_labels = []
    preds = []

    tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

    for i, (images, labels,_) in tk1:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            with autocast():
                y_preds = model(images.float())
                y_preds = y_preds.sigmoid()

        valid_labels.append(labels.to('cpu').numpy())
        preds.append(y_preds.to('cpu').numpy())
    preds = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)

    return preds,valid_labels



        

    




log = logging.getLogger(__name__)
@hydra.main(config_path="/home/abebe9849/Nploid/config",config_name="base")
def main(CFG : DictConfig) -> None:

    seed_torch(seed=CFG.general.seed)
    #os.environ["CUDA_VISIBLE_DEVICES"]=f"{CFG.general.device}"
    log.info(f"===============exp_num{CFG.general.exp_num}============")

    folds = pd.read_csv("/home/abebe9849/Nploid/select_5fold.csv")
    #folds = pd.read_csv("/home/u094724e/aimed2022/src/outputs/2022-03-14/15-40-59/oof.csv")
    #if CFG.general.debug:folds = folds.sample(len(folds)//10).reset_index(drop=True)
    preds = []
    valid_labels = []
    oof = pd.DataFrame()
    
    if CFG.psuedo_label!=0:
        test_pl=pd.read_csv(CFG.psuedo_label)
        test_pl["fold"]=999

        test_pl["pneumonia"]=test_pl["pneumonia"]
        test_pl["file_path"]="/home/u094724e/aimed2022/AImed_data/image/image/"+test_pl["id"]+".png"

    else:
        test_pl = 0
        



    for fold in range(5):
        _preds, _valid_labels,_oof_val = train_fn(CFG,fold,folds,test_pl)
        preds.append(_preds)
        valid_labels.append(_valid_labels)
        oof = pd.concat([oof,_oof_val])
    preds = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)

    log.info(f"OOF")
    AUC_score = roc_auc_score(valid_labels, preds)
    log.info(f"AUC_score  {AUC_score}")
    oof.to_csv(f"oof_{CFG.general.exp_num}.csv",index=False)




if __name__ == "__main__":
    main()
