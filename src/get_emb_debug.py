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
        imgs = glob.glob(f"{image_id}*tif")
        imgs =  np.stack([self.transform(image=cv2.imread(img)[:,:,::-1])['image']  for img in imgs])
        image = torch.from_numpy(imgs.transpose(0,3,1,2)).float()

        label = self.labels[idx]

        
        return image, torch.tensor(label).float()

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

    torch.cuda.set_device(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"### fold: {fold} ###")

    val_idx = folds[folds['fold'] == fold].index
    val_folds = folds.loc[val_idx].reset_index(drop=True)

    valid_dataset = TrainDataset(val_folds,train=False,
                                 transform1=get_transforms(data='valid',CFG=CFG),CFG=CFG)#

    valid_loader = DataLoader(valid_dataset, batch_size=CFG.train.batch_size, shuffle=False, num_workers=8,pin_memory=True)

    ###  model select ============
    model = Model_iafoss(base_model=CFG.model.name)
    weights_path = f"/home/abebe9849/Nploid/src/outputs/2022-08-27/17-12-49/fold{fold}_iafoss000_best_AUC.pth"
    state_dict = torch.load(weights_path,map_location=device)
    model.load_state_dict(state_dict)
    model.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten())
    model = model.to(device)


    # ============



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
        
    for epoch in range(1):

        model.eval()

        LOGITS = []
        embs = []
        tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

        for i, (images, _) in tk1:
            images = images.to(device)

            with torch.no_grad():
                with autocast():
                    emb = model(images)

            embs.append(emb.detach())


        embs = torch.cat(embs).cpu().numpy()


    return embs



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
@hydra.main(config_path="/home/abebe9849/Nploid/src/outputs/2022-08-27/17-12-49/.hydra/",config_name="config")
def main(CFG : DictConfig) -> None:

    seed_torch(seed=CFG.general.seed)
    #os.environ["CUDA_VISIBLE_DEVICES"]=f"{CFG.general.device}"
    log.info(f"===============exp_num{CFG.general.exp_num}============")

    folds = pd.read_csv("/home/abebe9849/Nploid/select_5fold_iafoss.csv")
    #folds = pd.read_csv("/home/u094724e/aimed2022/src/outputs/2022-03-14/15-40-59/oof.csv")
    #if CFG.general.debug:folds = folds.sample(len(folds)//10).reset_index(drop=True)
    preds = []
    valid_labels = []
    oof = pd.DataFrame()


    for fold in range(5):
        _embs = train_fn(CFG,fold,folds,0)
        preds.append(_embs)

    embs = np.concatenate(preds)
    print(embs.shape,folds.shape)

    np.save("/home/abebe9849/Nploid/src/outputs/2022-08-27/17-12-49/emb.npy",embs)




if __name__ == "__main__":
    main()
