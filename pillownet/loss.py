from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf


def smooth_l1(sigma=3.0):
    return None


def dice_coef(y_true, y_pred, smooth=1):
    intersect = K.sum(y_true * y_pred)
    denom = K.sum(y_true + y_pred)
    return K.mean((2. * intersect / (denom + smooth)))


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def bce_dice_loss(bce_weight=1.0, dice_weight=1.0):
    def bce_dice_loss_fixed(y_true, y_pred):
        return bce_weight * binary_crossentropy(y_true, y_pred) + dice_weight * dice_coef_loss(y_true, y_pred)
    return bce_dice_loss_fixed


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


def binary_crossentropy_cut(y_true, y_pred):
    mask_0 = K.equal(y_true, 0)
    mask_1 = K.equal(y_true, 1)
    mask = tf.logical_or(mask_0, mask_1)
    losses = K.binary_crossentropy(y_true, y_pred)
    losses = tf.boolean_mask(losses, mask)
    masked_loss = K.mean(losses)
    return masked_loss


def binary_accuracy_cut(y_true, y_pred):
    mask_0 = K.equal(y_true, 0)
    mask_1 = K.equal(y_true, 1)
    mask = tf.logical_or(mask_0, mask_1)
    equals = K.equal(y_true, K.round(y_pred))
    equals = tf.boolean_mask(equals, mask)
    masked_accuracy = K.mean(equals)
    return masked_accuracy
