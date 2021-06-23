import numpy as np
import os
import random
import glob
from natsort import natsorted

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(7)
rn.seed(7)

#folder structure
#dresden_256_patches --> train, valid, test 
#train --> unalt, jpeg70, jpeg80, jpeg90, ... ... .. 
#valid --> unalt, jpeg70, jpeg80, jpeg90, ... ... .. 
#test --> unalt, jpeg70, jpeg80, jpeg90, ... ... .. 
#inside the unalt, jpeg70, ... ... ... subfolders there are 18 folders for each camera model and inside the folders we have our images.

models = natsorted(os.listdir('/data/tianlei_store/RemNet/dresden_256_patches/train/unalt/'))

train_imdir=[]
train_label=[]

val_imdir=[]
val_label=[]
    
train_dir_dresden = '/data/tianlei_store/RemNet/dresden_256_patches/train/*/'

for i in range(len(models)):

    images_all = glob.glob(train_dir_dresden + models[i] + '/*')
    random.shuffle(images_all)
    train_imdir=train_imdir+images_all
    
    label_train=[i]*len(images_all)
    train_label=train_label+label_train
    
val_dir_dresden = '/data/tianlei_store/RemNet/dresden_256_patches/valid/*/'

for i in range(len(models)):

    images_all = glob.glob(val_dir_dresden + models[i] + '/*')
    random.shuffle(images_all)
    val_imdir=val_imdir+images_all
    
    label_val = [i]*len(images_all)
    val_label=val_label+label_val