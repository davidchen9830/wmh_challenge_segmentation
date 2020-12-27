import numpy as np

import matplotlib.pyplot as plt

import skimage.io

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, Concatenate, Cropping2D
from tensorflow.keras.optimizers import Adam

from utils import pad, get_slices, reconstruct

def get_model(shape, channels):
    inputs = Input(shape=(*shape, channels))
    
    def down_layers(inputs, filters):
        conv_1 = Conv2D(filters=filters, kernel_size=3, activation='relu')(inputs)
        conv_2 = Conv2D(filters=filters, kernel_size=3, activation='relu')(conv_1)
        pool = MaxPool2D()(conv_2)
        return pool, conv_2
    
    def up_layers(inputs, cropped, filters):
        conv_tr = Conv2DTranspose(filters=filters, kernel_size=2, strides=2)(inputs)
        concat = Concatenate()([conv_tr, cropped])
        conv_1 = Conv2D(filters=filters, kernel_size=3, activation='relu')(concat)
        conv_2 = Conv2D(filters=filters, kernel_size=3, activation='relu')(conv_1)
        return conv_2
    
    pool_1, conv_1_2 = down_layers(inputs, 64)
    pool_2, conv_2_2 = down_layers(pool_1, 128)
    pool_3, conv_3_2 = down_layers(pool_2, 256)
    pool_4, conv_4_2 = down_layers(pool_3, 512)
    
    
    conv_5_1 = Conv2D(filters=1024, kernel_size=3, activation='relu')(pool_4)
    conv_5_2 = Conv2D(filters=1024, kernel_size=3, activation='relu')(conv_5_1)
    
    conv_6_2 = up_layers(conv_5_2, Cropping2D(cropping=(4, 4))(conv_4_2), 512)
    conv_7_2 = up_layers(conv_6_2, Cropping2D(cropping=(16, 16))(conv_3_2), 256)
    conv_8_2 = up_layers(conv_7_2, Cropping2D(cropping=(40, 40))(conv_2_2), 128)
    conv_9_2 = up_layers(conv_8_2, Cropping2D(cropping=(88, 88))(conv_1_2), 64)
    
    outputs = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(conv_9_2)
    model = Model(inputs=inputs, outputs=outputs, name="unet")
    
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy')
    
    return model