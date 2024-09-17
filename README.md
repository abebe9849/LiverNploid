# LiverNploid
[[paper]()]

Official implementation of the paper "Selective identification of polyploid hepatocellular carcinomas with poor prognosis by artificial intelligence-based pathological image recognition".

######## be going to attach figure 1 here after it is adopted.


### Key points


## Requirements
- **Operating system**: Testing has been performed on Ubuntu 20.04.
- Python == 3.9
- PyTorch == 1.12.0

### pretraining

```
cd dino
python -m torch.distributed.launch --nproc_per_node=2 main_dino.py --arch vit_small --batch_size_per_gpu 256
```
If you experience NaN values in the DINO loss, please set `fp_16` to `False`, and also reduce the value of the gradient clipping.

### train efficeintnet_b0

python src/debugDINO_aug.py
python src/debug_inf_aug_TGCA.py

### train efficeintnet_b0_gray
python src/debug_aug_gray.py

### train dino pretrained model
python src/debugDINO_aug.py
python src/debug_inf_aug_DINO_TCGA.py

#### TCGA's data anottation
preprocess/TCGA_GT.csv 

#### plot tsne image
python src/tsne__hepato.py
python src/tsne_survivail.py
python src/tsne_survivail_agg.py

#### plot KM curve

python preprocess/plot_KM.py


### Issues
Please open new issue threads specifying the issue with the codebase or report issues directly to masamasa20001002@gmail.com . 

### Citation


### License

The source code for the site is licensed under the MIT license

