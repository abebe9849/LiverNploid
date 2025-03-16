import torch
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import matplotlib.pyplot as plt
from pathlib import Path
import cellseg_models_pytorch
print(cellseg_models_pytorch.__file__)

from cellseg_models_pytorch.inference import SlidingWindowInferer
from cellseg_models_pytorch.utils import FileHandler
from src.unet import get_seg_model, convert_state_dict, MODEL_PARTS
valid_img_dir = "/data/RSNA/Nploid_test/valid_img_dir"
save_dir = "/data/RSNA/Nploid_test/save_dir4"
if not os.path.exists(valid_img_dir):
    os.makedirs(valid_img_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

 # SET THIS WHERE YOU WANT TO SAVE RESULTING MASKS
import glob
import shutil


# copy glob.glob("/home/share/dataset/HeadNeckSCC/dent1/256patch/*")[:50] to valid_img_dir
if 1:
    import pandas as pd
    import cv2,tqdm

    #df = pd.read_csv("/home/abebe9849/Nploid/HEIP/extra.csv")
    #source_files = df["file_path"].tolist()[:]
    source_files = ["AA.tif","BB.tif",...."CC.tif"]

    # Copy files
    for file in tqdm.tqdm(source_files):
        if os.path.exists(os.path.join(valid_img_dir, os.path.basename(file).replace(".tif",".jpg"))):
            continue
        img = cv2.imread(file)
        cv2.imwrite(os.path.join(valid_img_dir, os.path.basename(file).replace(".tif",".jpg")), img)
        #shutil.copy2(file, valid_img_dir)




unet = get_seg_model()
#kaggle datasets download abebe9849/hepatopretrainedinsseg にてダウンロード
ckpt_old = torch.load("/home/abebe9849/Nploid/HEIP/last.ckpt", map_location=lambda storage, loc: storage)
new_state_dict = convert_state_dict(
    MODEL_PARTS,
    unet.state_dict(),
    ckpt_old["state_dict"]
)
unet.load_state_dict(new_state_dict, strict=True)


# First we run sliding window inference. 
# The 20 validation results are saved in inferer.out_masks class variable
# NOTE: The runtime and segmentation performance depend heavily on the 
# `stride`` parameter. Here we use a small stride to get the best segmentation perf
# but this sacrifices the performance a little. Also saving to disk is slow.
inferer = SlidingWindowInferer(
    unet,
    valid_img_dir,
    out_activations={"inst": "softmax", "type": "softmax", "omnipose": None},
    out_boundary_weights={"inst": False, "type": False, "omnipose": True},
    patch_size=(256, 256),
    stride=80,
    padding=120,
    instance_postproc="omnipose",
    batch_size=32,
    save_dir=save_dir
)

inferer.infer()