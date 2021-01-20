import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Concatenate, Cropping2D, UpSampling2D
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from metrics import dice_coef, dice_coef_training


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2])
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)

def get_model(shape, channels):
    inputs = Input(shape=(*shape, channels))

    def down_layers(inputs, filters, ks):
        conv_1 = Conv2D(filters=filters, kernel_size=ks, padding='same', activation='relu')(inputs)
        conv_2 = Conv2D(filters=filters, kernel_size=ks, padding='same', activation='relu')(conv_1)
        pool = MaxPool2D(pool_size=(2, 2))(conv_2)
        return pool, conv_2

    def up_layers(inputs, conv, filters):
        conv_tr = UpSampling2D(size=(2, 2))(inputs)
        ch, cw = get_crop_shape(conv, conv_tr)
        cropped = Cropping2D((ch, cw))(conv)
        concat = Concatenate()([conv_tr, cropped])
        conv_1 = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same')(concat)
        conv_2 = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same')(conv_1)
        return conv_2

    pool_1, conv_1_2 = down_layers(inputs, 64, (5, 5))
    pool_2, conv_2_2 = down_layers(pool_1, 96, (3, 3))
    pool_3, conv_3_2 = down_layers(pool_2, 128, (3, 3))
    pool_4, conv_4_2 = down_layers(pool_3, 256, (3, 3))

    conv_5_1 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool_4)
    conv_5_2 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv_5_1)

    conv_6_2 = up_layers(conv_5_2, conv_4_2, 256)
    conv_7_2 = up_layers(conv_6_2, conv_3_2, 128)
    conv_8_2 = up_layers(conv_7_2, conv_2_2, 96)
    conv_9_2 = up_layers(conv_8_2, conv_1_2, 64)

    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(conv_9_2)
    model = Model(inputs=inputs, outputs=outputs, name='unet')
    model.compile(optimizer=Adam(lr=1e-4), loss=BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), dice_coef])
    return model
