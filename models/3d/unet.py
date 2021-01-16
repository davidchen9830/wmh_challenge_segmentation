import numpy as np

import matplotlib.pyplot as plt

import skimage.io

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Conv3D, MaxPool3D, Conv3DTranspose, Concatenate, Cropping3D, UpSampling3D, ZeroPadding3D
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from utils import pad, get_slices, reconstruct

def recall(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = tp / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*((p*r)/(p+r+K.epsilon()))

def weighted_binary_crossentropy(y_true, y_pred):
    b_ce = K.binary_crossentropy(y_true, y_pred)
    one_weight = 0.99
    zero_weight = 0.01

    weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
    weighted_b_ce = weight_vector * b_ce
    return K.mean(weighted_b_ce)

smooth = 1

def dice_coef_for_training(y_true, y_pred):
    print(np.shape(y_pred))
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    print(np.shape(y_pred))
    print(np.shape(y_true))
    return -dice_coef_for_training(y_true, y_pred)

def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2])
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    cc = (target.get_shape()[3] - refer.get_shape()[3])
    assert(cc >= 0)
    if cc % 2 != 0:
        cc1, cc2 = int(cc/2), int(cc/2) + 1
    else:
        cc1, cc2 = int(cc/2), int(cc/2)

    return (ch1, ch2), (cw1, cw2), (cc1, cc2)

def get_model(shape, channels):
    inputs = Input(shape=(*shape, channels))

    def down_layers(inputs, filters, ks):
        conv_1 = Conv3D(filters=filters, kernel_size=ks, padding='same', activation='relu')(inputs)
        conv_2 = Conv3D(filters=filters, kernel_size = ks, padding='same', activation='relu')(conv_1)
        pool = MaxPool3D(pool_size=(2,2,2))(conv_2)
        return pool, conv_2

    def up_layers(inputs, conv, filters):
        conv_tr = UpSampling3D(size=(2,2,2))(inputs)
        ch, cw, cc = get_crop_shape(conv, conv_tr)
        cropped = Cropping3D((ch, cw, cc))(conv)
        concat = Concatenate()([conv_tr, cropped])
        conv_1 = Conv3D(filters=filters, kernel_size=(3,3,3), activation='relu', padding='same')(concat)
        conv_2 = Conv3D(filters=filters, kernel_size=(3,3,3), activation='relu', padding='same')(conv_1)
        return conv_2

    pool_1, conv_1_2 = down_layers(inputs, 64, (5,5,3))
    pool_2, conv_2_2 = down_layers(pool_1, 96, (3,3,3))
    pool_3, conv_3_2 = down_layers(pool_2, 128, (3,3,3))
    pool_4, conv_4_2 = down_layers(pool_3, 256, (3,3,3))

    conv_5_1 = Conv3D(512, (3,3,3), padding='same', activation='relu')(pool_4)
    conv_5_2 = Conv3D(512, (3,3,3), padding='same', activation='relu')(conv_5_1)

    conv_6_2 = up_layers(conv_5_2, conv_4_2, 256)
    conv_7_2 = up_layers(conv_6_2, conv_3_2, 128)
    conv_8_2 = up_layers(conv_7_2, conv_2_2, 96)
    conv_9_2 = up_layers(conv_8_2, conv_1_2, 64)

    ch, cw, cc =  get_crop_shape(inputs, conv_9_2)
    zero_pad = ZeroPadding3D(padding=(ch, cw, cc))(conv_9_2)
    outputs = Conv3D(1, (1,1,1), padding='same', activation='sigmoid')(zero_pad)
    model = Model(inputs=inputs, outputs=outputs, name='unet')
    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), dice_coef_for_training])
    return model