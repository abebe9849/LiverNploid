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
from code_factory.cam import *


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
        imgs = skimage.util.view_as_blocks(imgs, (256, 256, 3)).squeeze()#8,8,256,256,3

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
            input1 = self.model.forward_features(input1)#bs*num_tile,embed_dim,h,w
            shape = input1.size()
            input1 = input1.view(-1,n,shape[1],shape[2],shape[3])
            input1 = input1.permute(0,2,1,3,4).contiguous().view(-1,shape[1],shape[2]*n,shape[3])
            y = self.head(input1)
            """
            x =  self.model(input1)
            
            embeds, _ = self.gru(x.view(batch_size,n,x.shape[1]))
            embeds = self.pool(embeds.permute(0,2,1))[:,:,0]
            y = self.exam_predictor(embeds)
            """
            return y

#model = Model_iafoss("dino_vit_s")


#### model ================


def reverse_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    x[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
    return x

import torch.nn.functional as F
def visualize(img, cam):
    """
    Synthesize an image with CAM to make a result image.
    Args:
        img: (Tensor) shape => (1, 3, H, W)
        cam: (Tensor) shape => (1, 1, H', W')
    Return:
        synthesized image (Tensor): shape =>(1, 3, H, W)
    """

    _, _, H, W = img.shape
    cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
    
    cam = 255 * cam.squeeze()
    heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))
    heatmap = heatmap.float() / 255
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap*0.3 + img.cpu()*0.7
    result = result.div(result.max())

    return result


def patch2mid_tiff(imgs):
    #64,256,2,56,3→2048*2048*3
    imgs = np.stack([imgs[:8],imgs[8*1:8*2],imgs[8*2:8*3],imgs[8*3:8*4]
                  ,imgs[8*4:8*5],imgs[8*5:8*6],imgs[8*6:8*7],imgs[8*7:]
                  ])
    imgs = np.concatenate(imgs,axis=1)
    imgs = np.concatenate(imgs,axis=1)
    return imgs


def inf_func(model, valid_loader, device,file_names,DIR):
    #models = [m for m in models]
    model.eval()
    DIR = DIR+"/CAM"
    
    os.makedirs(DIR,exist_ok=True)


    target_layer = model.model.conv_head
    #target_layer = model.model.layer4[2].conv3 #for resnet

    model = GradCAM(model, target_layer)
    #model.model.classifier=nn.Linear(1280,9)
    preds = []

    tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

    for i, images in tk1:
        if i<186:continue
        images = images.to(device,non_blocking=True)
        #print(images.shape)
        with autocast():
            cam, _,prob= model(images)
            with torch.no_grad():
                images = reverse_normalize(images.detach().to('cpu').squeeze())

                cam = cam.detach().to('cpu')#.squeeze()
                cam = np.uint8((255 * F.interpolate(cam,mode='bilinear',scale_factor=(256/8))).squeeze())

                cam = np.stack([cv2.applyColorMap(c, cv2.COLORMAP_JET) for c in cam])[:,:,:,::-1].transpose(0,3,1,2).copy()
                cam = torch.from_numpy(cam)
                cam = cam.float() / 255

                cam = cam*0.2 + images*0.8
                #result = images
                cam = cam.div(cam.max()).numpy().transpose(0,2,3,1)
                cam = patch2mid_tiff(cam)
                plt.imshow(cam)
                c = file_names[i]
                plt.savefig(f"{DIR}/{c}_{prob}.png")
                print("result",cam.shape)
            
            
            
            
            
            
                #print(y_preds)

        preds.append(cam)
    preds = torch.cat(preds).to('cpu').numpy()

    return preds



def submit(CFG,test,DIR):
    torch.cuda.set_device(CFG.general.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = []
    for i in  [0]:
        model = Model_iafoss(base_model=CFG.model.name).to(device)
        #if i==3:continue #AI049
        model.load_state_dict(torch.load(f"{DIR}/fold{i}_nosiy__best_AUC.pth", map_location="cpu"))
        #models.append(model)
        
    valid_dataset = TestDataset(test,train=False,
                                 transform1=get_transforms(data='valid',CFG=CFG),CFG=CFG)# 
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=12,pin_memory=True)
    file_names = [p.split("/")[-1].replace(".tif","").replace(" ","") for p in test["file_path"].to_numpy()]

    tets_preds = inf_func(model, valid_loader, device,file_names,DIR)
    
    test["pred"]=tets_preds
    
    return test
    
    

def submit_folds(CFG,folds,DIR):
    torch.cuda.set_device(CFG.general.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    oof = pd.DataFrame()
    for i in   range(5):
        model = Model_iafoss(base_model=CFG.model.name).to(device)
        #if i==3:continue #AI049

        model.load_state_dict(torch.load(f"{DIR}/fold{i}_nosiy__best_AUC.pth", map_location="cpu"))
        
        val_folds = folds[folds['fold'] == i].reset_index(drop=True)
        valid_dataset = TestDataset(val_folds,train=False,
                                 transform1=get_transforms(data='valid',CFG=CFG),CFG=CFG)# 
        valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=12,pin_memory=True)
        test_preds = val_func(model, valid_loader, device)
    
        val_folds["pred"]=test_preds
        oof = pd.concat([oof,val_folds])
    
    return oof
        

    


"""
df = pd.DataFrame()
df["file_path"]=glob.glob("/data/RSNA/Nploid_test/20221216多倍体データ/20221120_MidTIFF/*/*")[:]
path_  = df["file_path"].values[0]
def func(x):
    x = x.split("/")[-2].split("_")[2]
    return "AI"+x
df["patient"]= np.vectorize(func)(df["file_path"])


new_df = []

for p in df["patient"].unique():
    tmp = df[df["patient"]==p].reset_index(drop=True).sample(n=5,random_state=2023)
    new_df.append(tmp)
    
df  = pd.concat(new_df).reset_index(drop=True)

df.to_csv("/home/abebe9849/Nploid/test_for_CAM.csv",index=False)
"""

df = pd.read_csv("/home/abebe9849/Nploid/test_for_CAM.csv")
    



#df  = pd.read_csv("/home/abebe9849/Nploid/src/outputs/2022-12-16/23-06-02/test.csv")
#print(df.shape)
#df = df[df["pred"].isna()].reset_index(drop=True)

DIR = "/home/abebe9849/Nploid/src/outputs/2023-01-06/_wo_ssr_add_brightcont"
DIR = "/home/abebe9849/Nploid/src/outputs/2023-01-07/b0_aug"
#DIR = "/home/abebe9849/Nploid/src/outputs/2023-02-28/resnet50d_select"
#DIR = "/home/abebe9849/Nploid/src/outputs/2023-02-27/resnet50d"
log = logging.getLogger(__name__)
@hydra.main(config_path=f"{DIR}/.hydra/",config_name="config")
def main(CFG : DictConfig) -> None:

    seed_torch(seed=CFG.general.seed)
    CFG.general.device = 0
    #os.environ["CUDA_VISIBLE_DEVICES"]=f"{CFG.general.device}"
    log.info(f"===============exp_num{CFG.general.exp_num}============")
    log.info(f"{DIR}")

    
    folds = pd.read_csv("/home/abebe9849/Nploid/all_5folds.csv")
    #oof_df =  submit_folds(CFG,folds,DIR)
    #oof_df.to_csv("oof.csv",index=False)
    
    #AUC_score = roc_auc_score(oof_df["label"], oof_df["pred"])
    #log.info(f"======CV  AUC{AUC_score}============")
    test_df = submit(CFG,df,DIR)
    
    test_df.to_csv("test.csv",index=False)

        






if __name__ == "__main__":
    main()
