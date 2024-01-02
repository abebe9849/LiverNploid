

import cupy as cp

from cuml import TSNE
from cuml import KMeans
from cuml.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


#from cuml import UMAP
#from sklearn.manifold import TSNE
import numpy as np
import os
import pandas as pd

#DIR = "/home/abebe9849/Nploid/src/outputs/2023-01-30/b0_aug_gray"

DIR = "/home/abebe9849/Nploid/src/outputs/2023-03-01/DINO_emb"
#DIR = "/home/abebe9849/Nploid/src/outputs/2023-02-27/resnet50d"
test =pd.read_csv("/home/abebe9849/Nploid/src/outputs/2023-01-07/b0_aug/test.csv")
df = pd.read_csv("/home/abebe9849/Nploid/新岡先生共有_臨床情報_松浦追記_20230131.csv")
"""
重みを固定した層が少ないほど見た目が近い画像が近傍に埋め込まれている

"""




print(df["heptitis_type"].value_counts())
print(df.columns)



fold = 4
#exit()

import time
s = time.time()
RANDOM_STATE = 2023
perplexity = 10
clist = ["orange","pink","blue","brown","red","grey","yellow","green"]
clist = ["orange","blue"]
color_dict ={"orange":[255,165,0],"pink":[255,10,255],"blue":[0,0,255],
"brown":[153,51,0],"red":[255,0,0],"grey":[150,150,150],
"yellow":[255,255,0],"green":[0,125,0],"black":[0,0,0],
}
features = cp.load(f"{DIR}/test_{fold}_emb.npy")

X_reduced = TSNE(n_components=2, random_state=RANDOM_STATE,perplexity=perplexity).fit_transform(features)

cp.save(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}.npy",X_reduced)

X_reduced = cp.asnumpy(X_reduced)
#X_reduced = np.load("/home/abebe9849/Nploid/src/outputs/2023-01-07/b0_aug/test_0_tsne_p10_seed_2023.npy")
test["tsne0"] = X_reduced[:, 0]
test["tsne1"] = X_reduced[:, 1]



dict_ = dict(zip(df["AI No."],df["AFP"]))
def func(x):
    out = dict_[x]
    return out
test["AFP"]=np.log10(test["patient"].apply(func))

dict_2 = dict(zip(df["AI No."],df["heptitis_type"]))
def func2(x):
    out = dict_2[x]
    return out
test["heptitis_type"]=test["patient"].apply(func2)

dict_3 = dict(zip(df["AI No."],df["組織型"]))
def func3(x):
    out = dict_3[x]
    return out
test["組織型"]=test["patient"].apply(func3)



DIR = DIR+"/SAVE2"
os.makedirs(DIR,exist_ok=True)
    
plt.figure(figsize=(14,12))
cm = plt.cm.get_cmap('bwr')

plt.scatter(X_reduced[:, 0],X_reduced[:, 1], c=test["AFP"], vmin=test["AFP"].min(), vmax=test["AFP"].max(), s=6, cmap=cm)
plt.colorbar(aspect=30, pad=0.02)
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_AFP.png")





plt.figure(figsize=(20,20))

for i in test["patient"].unique():
    tmp = test[test["patient"]==i]

    x = tmp["tsne0"]
    y = tmp["tsne1"]
    plt.scatter(x,y, s=15,label=i)
    
plt.legend()
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_patientby.png")

plt.figure(figsize=(20,20))
clist = ["orange","green","blue","yellow","red","gray","brown","black","pink"]
cdict_heptitis_type = dict(zip(test["heptitis_type"].unique(),clist))
color_dict ={"orange":[255,165,0],"pink":[255,10,255],"blue":[0,0,255],"brown":[153,51,0],"red":[255,0,0],"gray":[150,150,150],"yellow":[255,255,0],"green":[0,125,0],"black":[0,0,0],}


for i in test["heptitis_type"].unique():
    tmp = test[test["heptitis_type"]==i]

    x = tmp["tsne0"]
    y = tmp["tsne1"]
    c_ = color_dict[cdict_heptitis_type[i]]
    c_ = [c/255. for c in c_]
    plt.scatter(x,y, s=15,label=i,c=c_)
    
plt.legend(fontsize=20)
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_heptitis_type.png")

plt.figure(figsize=(20,20))

cdict_histological = dict(zip(test["組織型"].unique(),clist))


for i in test["組織型"].unique():
    tmp = test[test["組織型"]==i]

    x = tmp["tsne0"]
    y = tmp["tsne1"]
    c_ = color_dict[cdict_histological[i]]
    c_ = [c/255. for c in c_]
    plt.scatter(x,y, s=15,label=i,c=c_)
    
plt.legend(fontsize=20)
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_histological.png")

"""
がぞうのちらばり

"""

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def imscatter(x, y, image_list, ax=None, zoom=0.5):
    if ax is None:
        ax = plt.gca()
    im_list = [OffsetImage(cv2.imread(p)[:,:,::-1], zoom=zoom) for p in image_list]
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, im in zip(x, y, im_list):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


from PIL import Image
import cv2
from functools import reduce
from skimage.transform import resize
def plot_tiles(imgs, emb, grid_units=50, pad=2):


    #imgs = [cv2.resize(cv2.imread(p)[:,:,::-1],(256,256)) for p in imgs]
    # roughly 1000 x 1000 canvas
    cell_width = 10000 // grid_units
    s = grid_units * cell_width

    nb_imgs = len(imgs)

    embedding = emb.copy()

    # rescale axes to make things easier
    min_x, min_y = np.min(embedding, axis=0)
    max_x, max_y = np.max(embedding, axis=0)

    embedding[:, 0] = s * (embedding[:, 0] - min_x) / (max_x - min_x)
    embedding[:, 1] = s * (embedding[:, 1] - min_y) / (max_y - min_y)

    canvas = np.ones((s, s, 3))
    
    img_idx_dict = {}

    for i in range(grid_units):
        for j in range(grid_units):

            idx_x = (j * cell_width <= embedding[:, 1]) & (embedding[:, 1] < (j + 1) * cell_width)
            idx_y = (i * cell_width <= embedding[:, 0]) & (embedding[:, 0] < (i + 1) * cell_width)

            points = embedding[idx_y & idx_x]

            if len(points) > 0:

                img_idx = np.arange(nb_imgs)[idx_y & idx_x][0]  # take first available img in bin
                #tile = imgs[img_idx]               
                
                #resized_tile = resize(tile, output_shape=(cell_width - 2 * pad, cell_width - 2 * pad, 3))
                #print(resized_tile.shape)
                
                
                #exit()
                y = j * cell_width
                x = i * cell_width

                #canvas[s - y - cell_width+pad:s - y - pad, x + pad:x+cell_width - pad] = resized_tile
                
                #img_idx_dict[img_idx] = (x, x + cell_width, s - y - cell_width, s - y)
                img_idx_dict[img_idx] = (x,y)

    return canvas, img_idx_dict

def plot_tiles2(imgs, emb, grid_units=50, pad=2):


    
    # roughly 1000 x 1000 canvas
    cell_width = 10000 // grid_units
    s = grid_units * cell_width
    #imgs = [cv2.resize(cv2.imread(p)[:,:,::-1],(cell_width - 2 * pad,cell_width - 2 * pad)) for p in imgs]

    nb_imgs = len(imgs)

    embedding = emb.copy()

    # rescale axes to make things easier
    min_x, min_y = np.min(embedding, axis=0)
    max_x, max_y = np.max(embedding, axis=0)

    embedding[:, 0] = s * (embedding[:, 0] - min_x) / (max_x - min_x)
    embedding[:, 1] = s * (embedding[:, 1] - min_y) / (max_y - min_y)

    canvas = np.ones((s, s, 3))
    
    img_idx_dict = {}

    for i in range(grid_units):
        for j in range(grid_units):

            idx_x = (j * cell_width <= embedding[:, 1]) & (embedding[:, 1] < (j + 1) * cell_width)
            idx_y = (i * cell_width <= embedding[:, 0]) & (embedding[:, 0] < (i + 1) * cell_width)

            points = embedding[idx_y & idx_x]

            if len(points) > 0:

                img_idx = np.arange(nb_imgs)[idx_y & idx_x][0]  # take first available img in bin
                tile = imgs[img_idx]     
                
                #resized_tile = resize(tile, output_shape=(cell_width - 2 * pad, cell_width - 2 * pad, 3))
                #print(resized_tile.shape)
                tile = cv2.resize(cv2.imread(tile)[:,:,::-1],(cell_width - 2 * pad,cell_width - 2 * pad))/255.
                
                
                #exit()
                y = j * cell_width
                x = i * cell_width

                canvas[s - y - cell_width+pad:s - y - pad, x + pad:x+cell_width - pad] = tile
                
                img_idx_dict[img_idx] = (x, x + cell_width, s - y - cell_width, s - y)
                #img_idx_dict[img_idx] = (x,y)

    return canvas, img_idx_dict


def color_func(y,x,color_array):
    
    return np.stack([np.full((y,x),color_array[0]),np.full((y,x),color_array[1]),np.full((y,x),color_array[2])],axis=-1)/255.
    

def plot_tiles3(test_df, emb, grid_units=50, pad=6):


    
    # roughly 1000 x 1000 canvas
    cell_width = 15000 // grid_units
    s = grid_units * cell_width
    #imgs = [cv2.resize(cv2.imread(p)[:,:,::-1],(cell_width - 2 * pad,cell_width - 2 * pad)) for p in imgs]

    imgs = test_df["file_path"].to_numpy()
    histological_ = test_df["組織型"].to_numpy()
    heptitis_type_ = test_df["heptitis_type"].to_numpy()
    
    
    nb_imgs = len(imgs)

    embedding = emb.copy()

    # rescale axes to make things easier
    min_x, min_y = np.min(embedding, axis=0)
    max_x, max_y = np.max(embedding, axis=0)

    embedding[:, 0] = s * (embedding[:, 0] - min_x) / (max_x - min_x)
    embedding[:, 1] = s * (embedding[:, 1] - min_y) / (max_y - min_y)

    canvas_patient= np.ones((s, s, 3))
    canvas_heptitis_type = np.ones((s, s, 3))
    canvas_histological = np.ones((s, s, 3))
    
    
    img_idx_dict = {}

    for i in range(grid_units):
        for j in range(grid_units):

            idx_x = (j * cell_width <= embedding[:, 1]) & (embedding[:, 1] < (j + 1) * cell_width)
            idx_y = (i * cell_width <= embedding[:, 0]) & (embedding[:, 0] < (i + 1) * cell_width)

            points = embedding[idx_y & idx_x]

            if len(points) > 0:

                img_idx = np.arange(nb_imgs)[idx_y & idx_x][0]  # take first available img in bin
                tile = imgs[img_idx]     
                heptitis_type = heptitis_type_[img_idx]
                histological_type = histological_[img_idx]
                
                
                #resized_tile = resize(tile, output_shape=(cell_width - 2 * pad, cell_width - 2 * pad, 3))
                #print(resized_tile.shape)
                tile = cv2.resize(cv2.imread(tile)[:,:,::-1],(cell_width - 2 * pad,cell_width - 2 * pad))/255.
                
                
                #exit()
                y = j * cell_width
                x = i * cell_width
                
                
                canvas_heptitis_type[s - y - cell_width+pad//2:s - y - pad//2, x + pad//2:x+cell_width - pad//2]=color_func(cell_width-pad,cell_width-pad,color_dict[cdict_heptitis_type[heptitis_type]])
                canvas_histological[s - y - cell_width+pad//2:s - y - pad//2, x + pad//2:x+cell_width - pad//2]=color_func(cell_width-pad,cell_width-pad,color_dict[cdict_histological[histological_type]])

                canvas_patient[s - y - cell_width+pad:s - y - pad, x + pad:x+cell_width - pad] = tile
                canvas_heptitis_type[s - y - cell_width+pad:s - y - pad, x + pad:x+cell_width - pad] = tile
                canvas_histological[s - y - cell_width+pad:s - y - pad, x + pad:x+cell_width - pad] = tile
                
                img_idx_dict[img_idx] = (x, x + cell_width, s - y - cell_width, s - y)
                #img_idx_dict[img_idx] = (x,y)

    return canvas_patient,canvas_histological,canvas_heptitis_type,img_idx_dict

"""
_, img_idx_dict = plot_tiles(test["file_path"].to_numpy(), X_reduced, grid_units=25)

img_idx_dict_df = pd.DataFrame()
img_idx_dict_df["key"] = [test["file_path"].to_numpy()[i].replace("/data/RSNA/Nploid_test/20221216多倍体データ","") for i in  img_idx_dict.keys()]
img_idx_dict_df["x"] = [(i[0]-5000)*(300/5000) for i in  img_idx_dict.values()]
img_idx_dict_df["y"] = [(i[1]-5000)*(300/5000) for i in  img_idx_dict.values()]

img_idx_dict_df.to_csv(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_img_position_dict.csv",index=False)
"""
print("SCAAT")


plt.figure(figsize=(35,35))
canvas,canvas_histological,canvas_heptitis_type,_ = plot_tiles3(test, X_reduced, grid_units=25)
plt.imshow(canvas)
plt.axis("off")

plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_IMG25.png")
plt.figure(figsize=(35,35))
plt.imshow(canvas_histological)
plt.axis("off")

plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_IMG_histological.png")
plt.figure(figsize=(35,35))
plt.imshow(canvas_heptitis_type)
plt.axis("off")

plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_IMG_heptitis_type.png")
exit()
canvas, _ = plot_tiles2(test["file_path"].to_numpy(), X_reduced, grid_units=25)
plt.imshow(canvas)
plt.axis("off")

plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_IMG.png")

plt.figure(figsize=(25,25))
canvas, _ = plot_tiles2(test["file_path"].to_numpy(), X_reduced, grid_units=30)
plt.imshow(canvas)
plt.axis("off")

plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_IMG30.png")

plt.figure(figsize=(25,25))
canvas, _ = plot_tiles2(test["file_path"].to_numpy(), X_reduced, grid_units=35)
plt.imshow(canvas)
plt.axis("off")

plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_IMG35.png")

