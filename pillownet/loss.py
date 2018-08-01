import keras
from keras import backend as K
import tensorflow as tf


def focal(y_true, y_pred, alpha=0.25, gamma=2.0):
    binary_crossentropy_loss = K.binary_crossentropy(y_true, y_pred)
    return None


def smooth_l1(sigma=3.0):
    return None


def dice_coef(y_true, y_pred):
    ss = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    mask=K.cast(K.greater_equal(y_true_f,-0.5),dtype='float32')
    intersection = K.sum(y_true_f * y_pred_f * mask)
    return (2. * intersection + ss) / (K.sum(y_true_f * mask) + K.sum(y_pred_f * mask) + ss)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def crossentropy_cut(y_true, y_pred):  # cross-entropy loss
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f= tf.clip_by_value(y_pred_f, 1e-7, (1. - 1e-7))
    mask=K.cast(K.greater_equal(y_true_f,-0.5),dtype='float32')
    out = -(y_true_f * K.log(y_pred_f)*mask + (1.0 - y_true_f) * K.log(1.0 - y_pred_f)*mask)
    out=K.mean(out)
    return out
