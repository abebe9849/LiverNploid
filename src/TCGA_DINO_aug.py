"""
TCGAのデータで学習


"""


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
class TrainDataset_(Dataset):
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
        imgs = cv2.resize(imgs, dsize=(2048,2048), interpolation=cv2.INTER_LINEAR)

        imgs = skimage.util.view_as_blocks(imgs, (256, 256, 3)).squeeze()
        imgs = np.concatenate(imgs,0)#64,256,256,3
        imgs =  np.stack([self.transform(image=img)['image']  for img in imgs])
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
import vision_transformer as vits
class Model_iafoss(nn.Module):
    def __init__(self, base_model='tf_efficientnet_b0_ns',freeze=1):
        super(Model_iafoss, self).__init__()
        self.base_model = "dino_vit_s"
        
        
        checkpoint_key = "teacher"
        pretrained_weights = "/home/abebe9849/Nploid/dino/exp000/checkpoint.pth"
        self.model = vits.__dict__["vit_small"](patch_size=16, num_classes=0)
        state_dict = torch.load(pretrained_weights, map_location="cpu")
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

import transformers
class transformer_HF(nn.Module):
    """
    huggingfaceのいろんなモデルのencoderだけ使う
    """
    def __init__(self,model_name="microsoft/deberta-v3-small",in_size=512,hidden_size=512):
        super(transformer_HF, self).__init__()
        self.config = transformers.AutoConfig.from_pretrained(model_name)
        self.config.num_hidden_layers=2
        self.config.hidden_size=hidden_size
        self.config.num_attention_heads=4
        self.config.intermediate_size = 256
        
        if "deberta-v3" in model_name:
            self.model = transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Encoder(self.config)
        else:
            self.model = transformers.models.deberta.modeling_deberta.DebertaEncoder(self.config)
        self.gru = nn.GRU(in_size, hidden_size//2, bidirectional=True, batch_first=True, num_layers=2)

    def forward(self,embedding_output):
        embedding_output = self.gru(embedding_output)[0]
        mask = torch.ones(embedding_output.shape[0],embedding_output.shape[1]).to(embedding_output.device)
        x = self.model(embedding_output,mask)["last_hidden_state"]
        return x
    
class AttentionPooling(nn.Module):
    def __init__(self, num_layers=64, hidden_size=384, hiddendim_fc=1024):
        super(AttentionPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_fc = hiddendim_fc
        self.dropout = nn.Dropout(0.1)

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float()
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float()

    def forward(self, all_hidden_states):

        out = self.attention(all_hidden_states)
        out = self.dropout(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q.to(h.device), h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0).to(h.device), v_temp).squeeze(2)
        return v


#exit()

class Model_HIPT(nn.Module):
    def __init__(self, base_model='tf_efficientnet_b0_ns',freeze=1):
        super(Model_HIPT, self).__init__()
        self.base_model = "dino_vit_s"
        
        
        checkpoint_key = "teacher"
        pretrained_weights = "/home/abebe9849/Nploid/dino/vit256_small_dino.pth"
        self.model = vits.__dict__["vit_small"](patch_size=16, num_classes=0)
        state_dict = torch.load(pretrained_weights, map_location="cpu")
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

        #self.HF_traans = transformer_HF(model_name="microsoft/deberta-v3-small",in_size=nc,hidden_size=512)
        nc*=64
        self.head = nn.Sequential(nn.Linear(nc,512),
                        nn.ReLU(), nn.Dropout(0.5),nn.Linear(512,1))
        self.atten_pool = AttentionPooling(hidden_size=384*n_last_blocks)
        self.exam_predictor = nn.Linear(512*2, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        #attn_output, attn_output_weights = multihead_attn(query, key, value)
        #self.exam_predictor = nn.Linear(512, 1)
        


        
    def forward(self, input1):
        shape = input1.size()
        batch_size = shape[0]
        n = shape[1]
        input1 = input1.view(-1,shape[2],shape[3],shape[4])

        intermediate_output = self.model.get_intermediate_layers(input1, self.n_last_blocks)
        x = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        x = x.view(batch_size,n,x.shape[1])
        embeds, _ = self.gru(x)
        #embeds = self.HF_traans(x.view(batch_size,n,x.shape[1]))
        embeds = self.pool(embeds.permute(0,2,1))[:,:,0]
        
        #embeds = self.atten_pool(x) not work
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
        #x = x.view(batch_size,x.shape[1]*n)
        #y = self.head(x)
        
        return y,embeds

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
    

    #else:
    train_dataset = TrainDataset(tra_folds,train=True, transform1=get_transforms(data='train',CFG=CFG),CFG=CFG)#get_transforms(data='train',CFG=CFG)
    valid_dataset = TrainDataset(val_folds,train=False,transform1=get_transforms(data='valid',CFG=CFG),CFG=CFG)#

        

    train_loader = DataLoader(train_dataset, batch_size=CFG.train.batch_size*6, shuffle=True, num_workers=8,pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.train.batch_size*4, shuffle=False, num_workers=8,pin_memory=True)

    ###  model select ============
    print(CFG.model.name)
    log.info(f"====unfreeze {CFG.model.freeze} layer===")
    
    if CFG.model.name=="HIPT":
        model = Model_HIPT(base_model="dino_vit_s",freeze=CFG.model.freeze).to(device)
    else:
        model = Model_iafoss(base_model="dino_vit_s",freeze=CFG.model.freeze).to(device)
        
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

def submit(CFG,test,DIR):
    torch.cuda.set_device(CFG.general.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = []
    for fold in   range(5):
        model = Model_HIPT(base_model="dino_vit_s",freeze=1).to(device)
        #if i==3:continue #AI049
        model.load_state_dict(torch.load("fold{fold}_{CFG.general.exp_num}__HIPT_best_AUC.pth", map_location="cpu"),strict=False)

        #model.load_state_dict(torch.load(f"{DIR}/fold{i}_nosiy___HIPT_best_AUC.pth", map_location="cpu"),strict=False)
        models.append(model)
        
    valid_dataset = TestDataset(test,train=False,
                                 transform1=get_transforms(data='valid',CFG=CFG),CFG=CFG)# 
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=12,pin_memory=True)
    tets_preds,embs = inf_func(models, valid_loader, device)
    
    test["pred"]=tets_preds
    
    
    
    return test,embs

        



    

"""

df_GT = pd.read_csv("/home/abebe9849/Nploid/preprocess/TCGA_GT.csv")


from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
skf = StratifiedKFold(n_splits=5,random_state=2022,shuffle=True)
sgkf = StratifiedGroupKFold(n_splits=6,random_state=1000,shuffle=True)




df_all = []

for csv_path,Sample_ID,label in tqdm(zip(df_GT["FILE_ID"].values,df_GT["Sample ID"].values,df_GT["多倍体癌"].values),total=df_GT.shape[0]):
        #csv_path = "/data/RSNA/TGCAliver/csvs_/458e0da2-9f00-4ddd-a216-3fd4c0f58376.csv"
    df = pd.read_csv(f"/data/RSNA/TGCAliver/csvs_/{csv_path}.csv")
    df["label"]=label
    df["Sample_ID"] = Sample_ID
    df_all.append(df)
    #csv_path = csv_path.split("/")[-1]
    
df_all = pd.concat(df_all).reset_index(drop=True)

for fold, ( _, val_) in enumerate(sgkf.split(X=df_all, y=df_all.label,groups=df_all.Sample_ID)):
    df_all.loc[val_ , "fold"] = fold
    val_df = df_all[df_all["fold"]==fold]
    print(val_df["label"].value_counts())

df_all.to_csv("/data/RSNA/TGCAliver/csvs_/TCGA_train.csv",index=False)
print(df_all.columns,df_all.shape)

"""






#log = logging.getLogger(__name__)
#@hydra.main(config_path="/home/abebe9849/Nploid/config/",config_name="base_iafoss")

DIR = "/home/abebe9849/Nploid/src/outputs/2023-01-08/HIPT_unfreeze1_aug"


log = logging.getLogger(__name__)
@hydra.main(config_path=f"{DIR}/.hydra/",config_name="config")
def main(CFG : DictConfig) -> None:

    

    CFG.general.device = 0
    CFG.model.freeze = 0
    

    #os.environ["CUDA_VISIBLE_DEVICES"]=f"{CFG.general.device}"
    log.info(f"===============exp_num{CFG.general.exp_num}============")


    
    #folds = pd.read_csv("/home/abebe9849/Nploid/all_5folds20230309_v2.csv")
    

    folds = pd.read_csv("/data/RSNA/TGCAliver/csvs_/TCGA_train.csv")

    folds = folds[folds["fold"]!=5].reset_index(drop=True)
    
    print(folds.shape)
    
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
        seed_torch(seed=CFG.general.seed)
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
    
    
        
    
    df_tmp = pd.DataFrame()
    df_tmp["file_path"]=glob.glob("/data/RSNA/Nploid_test/20221216多倍体データ/20221120_MidTIFF/*/*")

    path_  = df_tmp["file_path"].values[0]
    def func(x):
        x = x.split("/")[-2].split("_")[2]
        return "AI"+x
    df_tmp["WSI_ID"]= np.vectorize(func)(df_tmp["file_path"])



    df_tmp = df_tmp[~df_tmp["WSI_ID"].isin(oof["WSI_ID"].unique())].reset_index(drop=True)

    test_df,embs = submit(CFG,df_tmp,DIR)

    test_df.to_csv(f"test.csv",index=False)
    
    for FOLD in range(5):    
        np.save(f"test_{FOLD}_emb.npy",embs[FOLD])

    
    test_df.to_csv("test.csv",index=False)
    tset_g = test_df.groupby("patient")["pred"].describe().reset_index()
    tset_g.to_csv(f"test_groupby.csv",index=False)
    

        


if __name__ == "__main__":
    main()
