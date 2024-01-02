
import matplotlib.pyplot as plt


#from cuml import UMAP
#from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
DIR = "/home/abebe9849/Nploid/src/outputs/2023-01-06/_wo_ssr_add_brightcont/"
DIR = "/home/abebe9849/Nploid/src/outputs/2023-01-07/b0_aug"
DIR = "/home/abebe9849/Nploid/src/outputs/2023-01-30/b0_aug_gray"
DIR = "/home/abebe9849/Nploid/src/outputs/2023-01-08/HIPT_unfreeze1_aug"

test =pd.read_csv(f"{DIR}/test.csv")

print(test.columns)
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

X_reduced = np.load(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}.npy")
test["tsne0"] = X_reduced[:, 0]
test["tsne1"] = X_reduced[:, 1]

plt.figure(figsize=(20,20))

for i in test["patient"].unique():
    tmp = test[test["patient"]==i]

    x = tmp["tsne0"]
    y = tmp["tsne1"]
    plt.scatter(x,y, s=15,label=i)
    
plt.legend()
plt.savefig(f"{DIR}/test_{fold}_tsne_p{perplexity}_seed_{RANDOM_STATE}_patientby.png")