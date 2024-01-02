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
        self.image_ids =self.df['file_path'].to_numpy()
        self.labels =self.df['label'].to_numpy()
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        """
        if "AI017" in image_id or "AI053" in image_id:
            image_id = "/data/RSNA/Nploid/HE_for_AI_SmallTIFF/HE_for_AI_SmallTIFF/" +"OLD"+ image_id.split("HE_for_AI_SmallTIFF")[-1]
            imgs = glob.glob(f"{image_id}*tif")
        elif "AI020" in image_id or "AI023" in image_id:
            image_id = "/data/RSNA/Nploid/HE_for_AI_SmallTIFF/HE_for_AI_SmallTIFF/" +"mismatch"+ image_id.split("HE_for_AI_SmallTIFF")[-1]
            imgs = glob.glob(f"{image_id}*tif")
        else:
            imgs = glob.glob(f"{image_id}*tif")
        """
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

        imgs =  cv2.imread(self.image_paths[idx])[:,:,::-1]#int(2048*(137/250)),int(2048*(137/250)),3

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
        imgs = cv2.resize(imgs, dsize=(2048,2048), interpolation=cv2.INTER_LINEAR)

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
                A.JpegCompression(),
                A.Downscale(scale_min=0.1, scale_max=0.15),
                ], p=CFG.aug.compress.p),
            GridMask(
                num_grid=CFG.aug.GridMask.num_grid,p=CFG.aug.GridMask.p),
            A.CoarseDropout(max_holes=CFG.aug.CoarseDropout.max_holes, max_height=CFG.aug.CoarseDropout.max_height, max_width=CFG.aug.CoarseDropout.max_width, p=CFG.aug.CoarseDropout.p),
            Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
    elif data == 'gray':
        return Compose([
            #Resize(CFG.preprocess.size,CFG.preprocess.size),
            #A.augmentations.crops.transforms.CenterCrop(int(CFG.preprocess.size*0.9),int(CFG.preprocess.size*0.9)),
            A.ToGray(),
            #Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
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

class Model_iafoss(nn.Module):
    def __init__(self, base_model='tf_efficientnet_b0_ns',pool="avg",pretrain=True):
        super(Model_iafoss, self).__init__()
        self.base_model = base_model 
        """
        if self.base_model=="dino_vit_s":
            checkpoint_key = "teacher"
            pretrained_weights = "/home/abebe9849/MAYO/dino/exp000/checkpoint0320.pth"
            self.model = vits.__dict__["vit_small"](patch_size=16, num_classes=0)
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            for _, p in self.model.named_parameters():
                p.requires_grad = False
            for _, p in self.model.head.named_parameters():
                p.requires_grad = True

            freeze =1
            for n, p in self.model.blocks.named_parameters():
                if int(n.split(".")[0])>=(12-freeze):
                    p.requires_grad = True
            avgpool_patchtokens = 0
            n_last_blocks = 4
            self.n_last_blocks = n_last_blocks
            nc = self.model.embed_dim * (n_last_blocks + int(avgpool_patchtokens))
            self.gru = nn.GRU(nc, 512, bidirectional=True, batch_first=True, num_layers=2)

            nc*=64
            self.head = nn.Sequential(nn.Linear(nc,512),
                            nn.ReLU(), nn.Dropout(0.5),nn.Linear(512,1))
            self.exam_predictor = nn.Linear(512*2, 1)
            self.pool = nn.AdaptiveAvgPool1d(1)
        """
        if self.base_model=="dino_vit_s":
            print("not imprement")
        else:
            self.model = timm.create_model(self.base_model, pretrained=True, num_classes=0,in_chans=3)
            #self.model.conv_stem = nn.Conv2d(2, 32, kernel_size=3, padding=1, stride=1, bias=False)
            nc = self.model.num_features
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(nc,512),
                            nn.ReLU(), nn.Dropout(0.5),nn.Linear(512,1))

            self.gru = nn.GRU(nc, 512, bidirectional=True, batch_first=True, num_layers=2)
            self.exam_predictor = nn.Linear(512*2, 1)
            self.pool = nn.AdaptiveAvgPool1d(1)

        
    def forward(self, input1):
        shape = input1.size()
        batch_size = shape[0]
        n = shape[1]
        input1 = input1.view(-1,shape[2],shape[3],shape[4])

        if "dino" in self.base_model:
            intermediate_output = self.model.get_intermediate_layers(input1, self.n_last_blocks)
            x = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            embeds, _ = self.gru(x.view(batch_size,n,x.shape[1]))
            embeds = self.pool(embeds.permute(0,2,1))[:,:,0]
            y = self.exam_predictor(embeds)
            #x = x.view(batch_size,x.shape[1]*n)
            #y = self.head(x)
           
            return y
        else:
            #"""
            x = self.model.forward_features(input1)#bs*num_tile,embed_dim,h,w
            shape = x.size()
            x = x.view(-1,n,shape[1],shape[2],shape[3])
            x = x.permute(0,2,1,3,4).contiguous().view(-1,shape[1],shape[2]*n,shape[3])
            y = self.head(x)
            """
            x =  self.model(input1)
            
            embeds, _ = self.gru(x.view(batch_size,n,x.shape[1]))
            embeds = self.pool(embeds.permute(0,2,1))[:,:,0]
            y = self.exam_predictor(embeds)
            """
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
    model = Model_iafoss(base_model=CFG.model.name).to(device)
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
                with autocast():
                    logits = model(images)
                    loss =  criterion(logits,labels.view(-1,1))

            valid_labels.append(labels)
            LOGITS.append(logits.detach())
            avg_val_loss += loss.item() / len(valid_loader)
        preds = torch.sigmoid(torch.cat(LOGITS)).cpu().numpy().squeeze() 
        valid_labels = torch.cat(valid_labels).cpu().numpy()

        print(preds.shape,valid_labels.shape)
        print(valid_labels.mean(axis=0))

        #each_auc,score =AUC(true=valid_labels,predict=preds)
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



def inf_func(models, valid_loader, device):
    #models = [m for m in models]
    for model in models:
        model.eval()

    preds = []

    tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

    for i, images in tk1:
        images = images.to(device,non_blocking=True)

        with torch.no_grad():
            with autocast():
                y_preds = [m(images.float()).sigmoid() for m  in models]
                #print(y_preds)
                y_preds  = torch.stack(y_preds).mean(0)

        preds.append(y_preds)
    preds = torch.cat(preds).to('cpu').numpy()

    return preds


def submit(CFG,test,DIR):
    torch.cuda.set_device(CFG.general.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = []
    for fold in   range(5):
        model = Model_iafoss(base_model=CFG.model.name).to(device)
        #if i==3:continue #AI049
        model.load_state_dict(torch.load(f"{DIR}/fold{fold}_{CFG.general.exp_num}_best_AUC.pth", map_location="cpu"))
        models.append(model)
        
    if "gray" in DIR.split("/")[-1]:
        valid_dataset = TestDataset(test,train=False,transform1=get_transforms(data='gray',CFG=CFG),CFG=CFG)# 
    else:
        valid_dataset = TestDataset(test,train=False,transform1=get_transforms(data='valid',CFG=CFG),CFG=CFG)# 
    valid_loader = DataLoader(valid_dataset, batch_size=12, shuffle=False, num_workers=12,pin_memory=True)
    tets_preds = inf_func(models, valid_loader, device)
    
    test["pred"]=tets_preds
    
    return test
    
    


        

    



csvs = glob.glob("/data/RSNA/TGCAliver/csvs_/*csv")[::-1]#380

print(len(csvs))
#slide propaties https://www.kaggle.com/code/naotous/openslide-how-to-open-whole-slide-images


#df  = pd.read_csv("/home/abebe9849/Nploid/src/outputs/2022-12-16/23-06-02/test.csv")
#print(df.shape)
#df = df[df["pred"].isna()].reset_index(drop=True)

DIR = "/home/abebe9849/Nploid/src/outputs/2023-01-07/b0_aug"
DIR = "/home/abebe9849/Nploid/src/outputs/2023-02-28/densenet121_select"
tmp_351 = pd.read_csv("/home/abebe9849/Nploid/src/outputs/2023-03-12/HIPT_unfreeze1_aug_select/predict_tcga.csv")
tmp_351 = tmp_351["FILE_ID"].unique()

log = logging.getLogger(__name__)
@hydra.main(config_path=f"{DIR}/.hydra/",config_name="config")
def main(CFG : DictConfig) -> None:

    seed_torch(seed=CFG.general.seed)
    CFG.general.device = 2
    if CFG.general.device==0:
        CFG.general.device+=1
    #os.environ["CUDA_VISIBLE_DEVICES"]=f"{CFG.general.device}"
    log.info(f"===============exp_num{CFG.general.exp_num}============")


    
    df = pd.read_csv("/data/RSNA/TGCAliver/csvs_/a14c4110-b51d-4ae8-90f2-86575d38b6b5.csv")
    
    os.makedirs(DIR+"/TCGA",exist_ok=True)
    for csv_path in csvs:
        t = csv_path.split("/")[-1].split(".csv")[0]
        csv_path_ = csv_path.split("/")[-1]
        if  os.path.exists(DIR+"/TCGA/"+csv_path_):continue
        if t not in tmp_351:continue
        if "TCGA_train.csv" in csv_path:continue
        #csv_path = "/data/RSNA/TGCAliver/csvs_/458e0da2-9f00-4ddd-a216-3fd4c0f58376.csv"
        log.info(f"{csv_path}")
        df = pd.read_csv(csv_path)
        test_df = submit(CFG,df,DIR)
        test_df.to_csv(DIR+"/TCGA/"+csv_path_,index=False)
        






if __name__ == "__main__":
    main()
