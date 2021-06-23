import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import log_loss
import keras
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
import numpy as np
from imageio import imread
import math
from scipy import signal
import random

batchsize = 64
total_image = len(train_imdir)
img_rows = 64
img_cols = 64
classes = 18
inp_size = 256
channel = 3
    
combined_train = list(zip(train_imdir, train_label))
random.shuffle(combined_train)
    
train_imdir[:], train_label[:] = zip(*combined_train)
    
combined_val= list(zip(val_imdir, val_label))
random.shuffle(combined_val)
    
val_imdir[:], val_label[:] = zip(*combined_val)

def generate_processed_batch(inp_data,label,batch_size = 50):

    batch_images = np.zeros((batch_size, img_rows, img_cols, channel))
    batch_label = np.zeros((batch_size, classes))
    
    while 1:
        for i_data in range(0,total_image,batch_size):
            for i_batch in range(batch_size):
                if i_data + i_batch >= len(inp_data):
                    continue 
                img = imread(inp_data[i_data+i_batch])
                
                a = inp_size - img_rows
                d = np.random.randint(0,a)
                img = img[d:d+img_rows,d:d+img_rows,:]
                
                lab = np_utils.to_categorical(label[i_data+i_batch],classes)

                batch_images[i_batch] = img
                batch_label[i_batch] = lab
            if i_data + i_batch >= len(inp_data):
                    continue 

            yield batch_images, batch_label

from keras.callbacks import TensorBoard

yb = TensorBoard(log_dir='.\\logs', histogram_freq=0,  
          write_graph=True, write_images=True)
csv_logger = CSVLogger('training_RemNet_without_unalt_train_manip.log')
callbacks_list= [keras.callbacks.ModelCheckpoint(
        filepath='RemNet_without_unalt_train_manip.h5',
        monitor='val_loss', verbose=1, 
        save_best_only=True, save_weights_only=False, mode='min'
    ), 
            
            
    EarlyStopping(monitor='val_loss', patience=15, verbose=1, min_delta=1e-3),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, cooldown=1, 
                               verbose=1, min_lr=1e-7), yb, csv_logger
]

training_gen = generate_processed_batch(train_imdir, train_label, batchsize)
val_gen = generate_processed_batch(val_imdir,val_label, batchsize)
    



img_rows, img_cols = img_rows, img_rows # Resolution of inputs
channel = 3
num_classes = classes
batch_size = batchsize
nb_epoch = 50
n = batchsize


ADAM = Adam(lr=1e-03)
# sgd = SGD(lr=1e-03, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss = 'categorical_crossentropy', optimizer = ADAM,  metrics = ['accuracy'])

history = model.fit_generator(training_gen,steps_per_epoch=int(total_image/n),nb_epoch=nb_epoch,validation_data=val_gen,
                    validation_steps=int(len(val_imdir)/n),callbacks=callbacks_list,
                    initial_epoch=0)


import pickle
with open('/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)