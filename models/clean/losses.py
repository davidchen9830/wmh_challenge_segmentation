import tensorflow.keras.backend as K


def weighted_binary_crossentropy(y_true, y_pred):
    b_ce = K.binary_crossentropy(y_true, y_pred)
    one_weight = 0.99
    zero_weight = 0.01

    weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
    weighted_b_ce = weight_vector * b_ce
    return K.mean(weighted_b_ce)
