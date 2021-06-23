import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,GlobalMaxPooling1D,GlobalAveragePooling2D, Conv3D, MaxPooling3D
from keras import backend as K
import numpy as np
from keras.engine.topology import Input
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras import backend as K

import tensorflow as tf
from keras.optimizers import SGD, Adam
from keras.layers import Input, merge, ZeroPadding2D, concatenate, Add
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from keras.layers import PReLU
from keras.callbacks import LearningRateScheduler

from keras.utils import np_utils
import numpy as np
from imageio import imread
import math
from scipy import signal
import random
import os
import re
import glob
import tqdm
from keras import layers
from keras import models
from keras.constraints import max_norm
import warnings
import keras
warnings.filterwarnings('ignore')
from keras.optimizers import SGD, Adam

from keras import layers
from keras import models
from keras.constraints import max_norm
import warnings
import keras
warnings.filterwarnings('ignore')
from keras.optimizers import SGD

img_height, img_width, img_channels = 64, 64, 3
nb_channels = 3
eps = 1.1e-5
nb_filter = 64
concat_axis = 3
nb_dense_block=4
growth_rate=32
reduction=0.0
dropout_rate=0.0
weight_decay=1e-4
classes=1000 
weights_path=None
nb_layers = [6,12,24,16]
compression = 1.0 - reduction

def antirectifier(x):
    x = K.l2_normalize(x, axis=1)
    
    return x

def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
#    assert len(shape) == 2  # only valid for 2D tensors
#    shape[-1] *= 2
    return tuple(shape)


image_tensor = layers.Input(shape=(img_height, img_width, img_channels), name = 'image')
image_tensor_bn = layers.BatchNormalization(name = 'image_bn')(image_tensor)

y = layers.Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='same')(image_tensor_bn)
y = layers.BatchNormalization(name = 'bn1')(y)

ya = keras.layers.concatenate([image_tensor_bn, y])

y = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(ya)
y = layers.BatchNormalization()(y)


yb = keras.layers.concatenate([image_tensor_bn, y])

y = layers.Conv2D(3, kernel_size=(3, 3), strides=(1,1), padding='same')(yb)
y = layers.BatchNormalization(name = 'biyog_er_age1')(y)

yc = layers.subtract(([image_tensor_bn, y]), name = 'biyog1')

yc_bn = layers.BatchNormalization(name = 'biyog_bn1')(yc)


y = layers.Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding='same')(yc_bn)
y = layers.BatchNormalization(name = 'bn2')(y)

ya = keras.layers.concatenate([yc_bn, y])

y = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(ya)
y = layers.BatchNormalization()(y)


yb = keras.layers.concatenate([yc_bn, y])

y = layers.Conv2D(3, kernel_size=(3, 3), strides=(1,1), padding='same')(yb)
y = layers.BatchNormalization(name = 'biyog_er_age2')(y)

yd = layers.subtract(([yc_bn, y]), name = 'biyog2')
yd_bn = layers.BatchNormalization(name = 'biyog_bn2')(yd)


y = layers.Conv2D(256, kernel_size=(3, 3), strides=(1,1), padding='same')(yd_bn)
y = layers.BatchNormalization(name = 'bn3')(y)

ya = keras.layers.concatenate([yd_bn, y])

y = layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(ya)
y = layers.BatchNormalization()(y)


yb = keras.layers.concatenate([yd_bn, y])

y = layers.Conv2D(3, kernel_size=(3, 3), strides=(1,1), padding='same')(yb)
y = layers.BatchNormalization(name = 'biyog_er_age3')(y)

yc = layers.subtract(([yd_bn, y]), name = 'biyog3')


#yc_bn = layers.BatchNormalization(name = 'biyog_bn4')(yc)
#
#y = layers.Conv2D(512, kernel_size=(3, 3), strides=(1,1), padding='same')(yc_bn)
#y = layers.BatchNormalization(name = 'bn4')(y)
#
#ya = keras.layers.concatenate([yc_bn, y])
#
#y = layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(ya)
#y = layers.BatchNormalization()(y)
#
#
#yb = keras.layers.concatenate([yc_bn, y])
#
#y = layers.Conv2D(3, kernel_size=(3, 3), strides=(1,1), padding='same')(yb)
#y = layers.BatchNormalization(name = 'biyog_er_age4')(y)
#
#yc = layers.subtract(([yc_bn, y]), name = 'biyog4')


#yc = Activation('tanh', name = 'asol_loss')(z)

#y = keras.layers.ActivityRegularization(l1=0.0, l2=0.0)(y)
#
#yc = keras.layers.Lambda(lambda x: K.max(0,x))(yc)

conv1 = Conv2D(64, (7, 7), padding='same', strides = (2,2))(yc)
bn1 = BatchNormalization()(conv1)
act1 = PReLU()(bn1)
    
#pool1 = MaxPooling2D(pool_size=(3,3), strides = (2,2))(act1)
    
    #conc1 = concatenate([conv1, conv2], axis=concat_axis)
conv2 = Conv2D(128, (5, 5), padding='same', strides = (2,2))(act1)
bn2 = BatchNormalization()(conv2)
act2 = PReLU()(bn2)
#
#pool2 = MaxPooling2D(pool_size=(3,3), strides = (2,2))(act2)
#    
conv3 = Conv2D(256, (3, 3), padding='same', strides = (2,2))(act2)
bn3 = BatchNormalization()(conv3)
act3 = PReLU()(bn3)
#    
#pool3 = MaxPooling2D(pool_size=(3,3), strides = (2,2))(act3)
#    
conv4 = Conv2D(512, (2,2), padding='same', strides = (2,2))(act3)
bn4 = BatchNormalization()(conv4)
act4 = PReLU()(bn4)
#
#conv5 = Conv2D(256, (3,3), padding='same')(act4)
#bn5 = BatchNormalization()(conv5)
#act5 = PReLU()(bn5)
#
#conv6 = Conv2D(256, (3,3), padding='same')(act5)
#bn6 = BatchNormalization()(conv6)
#act6 = PReLU()(bn6)
#
#conv7 = Conv2D(256, (3,3), padding='same')(act6)
#bn7 = BatchNormalization()(conv7)
#act7 = PReLU()(bn7)
#
#conv8 = Conv2D(512, (1,1), padding='same')(act7)
#bn7 = BatchNormalization()(conv8)
#act7 = PReLU()(bn7)

pool4 = AveragePooling2D(pool_size=(4,4))(act4)
    
out = Conv2D(18, (1, 1), padding='same')(pool4)
out = Flatten()(out)
out = Activation('softmax')(out)


# model = models.Model(inputs=[image_tensor], outputs=[out])

model = keras.models.load_model('RemNet_single_filter.h5')


#%%

#model.summary()

import keras.backend as K
import numpy as np
from scipy.io import savemat

def get_layer_outputs(test_image,name):

    outputs    = [model.get_layer(name).output]          # all layer outputs
    comp_graph = [K.function([model.input]+ [K.learning_phase()], [output]) for output in outputs]  # evaluation functions

    # Testing
    layer_outputs_list = [op([test_image, 1.]) for op in comp_graph]
    layer_outputs = []

    for layer_output in layer_outputs_list:
        print(layer_output[0][0].shape, end='\n-------------------\n')
        layer_outputs.append(layer_output[0][0])

    return layer_outputs

def plot_layer_outputs(layer_outputs,no_of_filter):    

    x_max = layer_outputs.shape[0]
    y_max = layer_outputs.shape[1]
    n     = layer_outputs.shape[2]

    L = []
    for i in range(n):
        L.append(np.zeros((x_max, y_max)))

    for i in range(n):
        for x in range(x_max):
            for y in range(y_max):
                L[i][x][y] = layer_outputs[x][y][i]


    L = np.squeeze(L)

    print ('Shape of conv:', L.shape)
    
    n = no_of_filter
    n = int(np.ceil(np.sqrt(n)))
    
    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(x_max,x_max))
    for i in range(no_of_filter):
        ax = fig.add_subplot(n,n,i+1)
        ax.imshow(L[i], cmap = 'gray')
        ax.axis('off')

def plot_filter_weights(idx):    

    L = model.layers[idx].get_weights()
    L = np.squeeze(L)

    print ('Shape of conv:', L.shape)
    
    n = L.shape[-1]
#    n = int(np.ceil(np.sqrt(n)))
    
    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(L.shape[0],L.shape[0]))
    for i in range(L.shape[-1]):
        ax = fig.add_subplot(n,n,i+1)
        ax.imshow(L[:,:,0,i], cmap = 'gray')
        ax.axis('off')

from imageio import imread   
     
def get_img(filepath):
    img = imread(filepath)
    img_center = np.expand_dims(img, axis=0)
    return img_center

import matplotlib.pyplot as plt



def  layer_to_visualize(layer):
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])
    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img_to_visualize)
    convolutions = np.squeeze(convolutions)

    print ('Shape of conv:', convolutions.shape)
    
    n = convolutions.shape[-1]
    n = int(np.ceil(np.sqrt(n)))
    
    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(12,8))
    for i in range(convolutions.shape[-1]):
        ax = fig.add_subplot(n,n,i+1)
        ax.imshow(convolutions[i])
        ax.axis('off')



# Specify the layer to want to visualize
#layer_to_visualize(x_out)

# As convout2 is the result of a MaxPool2D layer
# We can see that the image has blurred since
# the resolution has reduced 
#layer_to_visualize(convout2)       
#  
#%%
idx=90
#filepath = val_imdir[idx]

#filepath = 'G:\\manip_code_create\\MANIP_DATA\\unalt\\Kodak_M1063\\Kodak_M1063_0_10004_4.png'
#filepath = 'G:\\manip_code_create\\MANIP_DATA\\unalt\\FujiFilm_FinePixJ50\\FujiFilm_FinePixJ50_0_7559_4.png'
filepath = 'Rollei_RCP-7325XS_0_42342_5.png'
#filepath = 'G:\\manip_code_create\\MANIP_DATA\\unalt\\Samsung_NV15\\Samsung_NV15_0_45322_18.png'
#filepath = 'G:\\manip_code_create\\WORST_20_PATCHES\\unalt\\Samsung_L74wide\\Samsung_L74wide_0_43536_7.png'

# choose any image to want by specifying the index
img_to_visualize = get_img(filepath)
img_to_visualize = img_to_visualize[np.newaxis, 0, :64, :64, :]


#model.load_weights('without_activation_model.h5')
#plt.figure()
#plt.imshow(img_to_visualize[0])
#plt.axis('off')
#plt.savefig('img2.eps', format='eps', dpi=300)     

#L = get_layer_outputs(img_to_visualize,'image_bn')
#savemat('image1_bn.mat', mdict = {'image' : L[0]})
#plot_layer_outputs(L[0],3)
#plt.savefig('img2_bn.eps', format='eps', dpi=300)

#L = get_layer_outputs(img_to_visualize,'biyog1')
#savemat('image1_res1.mat', mdict = {'image' : L[0]})
#plot_layer_outputs(L[0],3)
#plt.savefig('man1_res1.eps', format='eps', dpi=300)

#L = get_layer_outputs(img_to_visualize,'biyog2')
#savemat('image1_res2.mat', mdict = {'image' : L[0]})
#plot_layer_outputs(L[0],3)
#plt.savefig('man1_res2.eps', format='eps', dpi=300)

L = get_layer_outputs(img_to_visualize,'biyog3')
savemat('image1_res3.mat', mdict = {'image' : L[0]})
plot_layer_outputs(L[0],3)
# plt.savefig('man1_res3.eps', format='eps', dpi=300)

#plt.figure()
#plt.imshow(L[0])


#
#L = get_layer_outputs(img_to_visualize,'asol_loss')
#plot_layer_outputs(L[0],3)
#
#L = get_layer_outputs(img_to_visualize,'relu1')
#plot_layer_outputs(L[0],64)




