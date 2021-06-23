from keras.layers import Dense, Dropout, Flatten, Input, concatenate, subtract, Conv2D, PReLU, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D
from keras import models

img_height, img_width, img_channels = 64, 64, 3

image_tensor = Input(shape=(img_height, img_width, img_channels))
image_tensor_bn = BatchNormalization(name = 'image_bn')(image_tensor)

y = Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='same')(image_tensor_bn)
y = BatchNormalization(name = 'bn1')(y)

ya = concatenate([image_tensor_bn, y])

y = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(ya)
y = BatchNormalization()(y)


yb = concatenate([image_tensor_bn, y])

y = Conv2D(3, kernel_size=(3, 3), strides=(1,1), padding='same')(yb)
y = BatchNormalization(name = 'biyog_er_age1')(y)

yc = subtract(([image_tensor_bn, y]), name = 'biyog1')

yc_bn = BatchNormalization(name = 'biyog_bn1')(yc)


y = Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding='same')(yc_bn)
y = BatchNormalization(name = 'bn2')(y)

ya = concatenate([yc_bn, y])

y = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(ya)
y = BatchNormalization()(y)


yb = concatenate([yc_bn, y])

y = Conv2D(3, kernel_size=(3, 3), strides=(1,1), padding='same')(yb)
y = BatchNormalization(name = 'biyog_er_age2')(y)

yd = subtract(([yc_bn, y]), name = 'biyog2')
yd_bn = BatchNormalization(name = 'biyog_bn2')(yd)


y = Conv2D(256, kernel_size=(3, 3), strides=(1,1), padding='same')(yd_bn)
y = BatchNormalization(name = 'bn3')(y)

ya = concatenate([yd_bn, y])

y = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(ya)
y = BatchNormalization()(y)


yb = concatenate([yd_bn, y])

y = Conv2D(3, kernel_size=(3, 3), strides=(1,1), padding='same')(yb)
y = BatchNormalization(name = 'biyog_er_age3')(y)

yc = subtract(([yd_bn, y]), name = 'biyog3')


conv1 = Conv2D(64, (7, 7), padding='same', strides = (2,2))(yc)
bn1 = BatchNormalization()(conv1)
act1 = PReLU()(bn1)
    
conv2 = Conv2D(128, (5, 5), padding='same', strides = (2,2))(act1)
bn2 = BatchNormalization()(conv2)
act2 = PReLU()(bn2)
#    
conv3 = Conv2D(256, (3, 3), padding='same', strides = (2,2))(act2)
bn3 = BatchNormalization()(conv3)
act3 = PReLU()(bn3)
#    
conv4 = Conv2D(512, (2,2), padding='same', strides = (2,2))(act3)
bn4 = BatchNormalization()(conv4)
act4 = PReLU()(bn4)
#
pool4 = AveragePooling2D(pool_size=(4,4))(act4)
    
out = Conv2D(18, (1, 1), padding='same')(pool4)
out = Flatten()(out)
out = Activation('softmax')(out)

model = models.Model(inputs=[image_tensor], outputs=[out])

model.summary()
