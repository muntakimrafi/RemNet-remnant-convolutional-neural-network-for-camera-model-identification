
import math
import numpy as np
import os
import glob
import time
from imageio import imwrite, imread
from PIL import Image
from operator import itemgetter
import matplotlib.pyplot as plt
import re
start = time.time()
#%%Manipulations
import scipy.misc

def resize(img, scale):
    return scipy.misc.imresize(img, scale, interp='bicubic')

from skimage import exposure

def adjust_gamma(img, gamma):
    return exposure.adjust_gamma(img, gamma=gamma)

def compress(img,quality):
    imwrite('temp.jpg', img, quality=quality)
    return imread('temp.jpg')

def gauss_noise(img, mean, var):
    row, col, ch = img.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = img + gauss
    return noisy

#%%
# gamma = 1.2
scale = 2.0
# quality = 90
#mean, var = 0, 1
def patch_creator(single_image_dir,num_row = 256,num_col = 256):
    im = Image.open(img_name)
    single_image = np.array(im)
    
    single_image = resize(single_image, scale)
    # single_image = adjust_gamma(single_image, gamma)
    # single_image = compress(single_image, quality)
#    single_image = gauss_noise(single_image, mean, var)
    
    
    a , b = single_image.shape[0], single_image.shape[1]
    k , m = a//num_row, b//num_col
    ind_row = np.repeat(range(0,k*num_row,num_row), m)
    ind_col = np.tile(range(0,m*num_col,num_col), k)
    image_patches = [single_image[a1:a1+num_row,a2:a2+num_col,:] for (a1,a2) in zip(ind_row,ind_col)]    
    return image_patches


def find_quality(patches, sel_no = 32):
    alpha = 0.7
    beta = 4
    gamma = math.log(0.01)    
    Constant_1 = np.repeat(np.array([alpha]),3)*np.repeat(np.array([beta]),3)
    Constant_2 = np.repeat(np.array([1]),3)-np.repeat(np.array([alpha]),3)
    Constant_3 = np.repeat(np.array([1]),3)  
    zipped = []
    quality = []
    for i in patches:
        img = i/255.0
        chnl_mean = np.mean(img, axis=(0,1))    
        chnl_std = np.std(img, axis=(0,1), ddof = 1)
        part_1 = Constant_1*(chnl_mean - chnl_mean*chnl_mean)
        part_2 = Constant_2*(Constant_3 - np.exp(np.repeat(np.array([gamma]),3)*chnl_std))
        img_qulty = np.mean(part_1 + part_2)    
        quality.append(img_qulty)
    zipped = zip(quality, patches)
    zipped = sorted(zipped,key=itemgetter(0))
    best_patches = zipped[-sel_no:]
    return best_patches


def saveimg(wrt_dir, img_counter, zipped_patches):
    k = 0  
    for i in zipped_patches:  
        image_wrt_dir = wrt_dir  +'//{}_{}.tif'.format(img_counter, k)
        imwrite(image_wrt_dir,i[1])
        k += 1    

def sorted_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)


imreadpath = '/data/tianlei_store/RemNet/dresden_data/train/' 
model = os.listdir(imreadpath)

imwritepath = '/data/tianlei_store/RemNet/dresden_256_patches/train/resize_2.0/'

num_row = 256
num_col = 256
sel_no = 20

import tqdm

for m in range(len(model)):
    images = glob.glob(imreadpath + model[m] + '/*')
    images = sorted_nicely(images)
    file_wr_path = imwritepath+ model[m]+'/'
    
    if not os.path.exists(file_wr_path):
        os.makedirs(file_wr_path)
    print("\nMaking Patches \t Model: {}\t".format(model[m]))   
    img_no = 0
    for img_name in tqdm.tqdm(images):  #only 6000 images are being taken
        img_no += 1
        try :
            patches = patch_creator(img_name,num_row = num_row,num_col = num_col)
        except:
            continue
        selected_patches = find_quality(patches,sel_no = sel_no)
        k = 0  
        for img in selected_patches:  
            k += 1
            image_wrt_dir = file_wr_path  +'/{}_{}.png'.format(img_name.split(os.sep)[-1].split('.jpg')[0],k)
            imwrite(image_wrt_dir,img[1])
