from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf


def dice_loss(y_true, y_pred, smooth=1e-6):
    """ Loss function base on dice coefficient.

    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    smooth : float
        small real value used for avoiding division by zero error.

    Returns
    -------
    keras tensor
        tensor containing dice loss.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    answer = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return -answer


def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-10):
    """ Tversky loss function.

    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    alpha : float
        real value, weight of '0' class.
    beta : float
        real value, weight of '1' class.
    smooth : float
        small real value used for avoiding division by zero error.

    Returns
    -------
    keras tensor
        tensor containing tversky loss.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
    answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
    return -answer


def jaccard_coef_logloss(y_true, y_pred, smooth=1e-10):
    """ Loss function based on jaccard coefficient.

    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    smooth : float
        small real value used for avoiding division by zero error.

    Returns
    -------
    keras tensor
        tensor containing negative logarithm of jaccard coefficient.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    falsepos = K.sum(y_pred) - truepos
    falseneg = K.sum(y_true) - truepos
    jaccard = (truepos + smooth) / (smooth + truepos + falseneg + falsepos)
    return -K.log(jaccard + smooth)


def jaccard_loss(y_true, y_pred, smooth=1e-10):
    """ Loss function based on jaccard coefficient.

    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    smooth : float
        small real value used for avoiding division by zero error.

    Returns
    -------
    keras tensor
        tensor containing negative logarithm of jaccard coefficient.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    falsepos = K.sum(y_pred) - truepos
    falseneg = K.sum(y_true) - truepos
    jaccard = (truepos + smooth) / (smooth + truepos + falseneg + falsepos)
    return -jaccard


def dice_coef(y_true, y_pred, smooth=1):
    intersect = K.sum(y_true * y_pred)
    denom = K.sum(y_true + y_pred)
    return K.mean((2. * intersect / (denom + smooth)))


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def bce_dice_loss(bce_weight=1.0, dice_weight=1.0):
    def bce_dice_loss_fixed(y_true, y_pred):
        return bce_weight * binary_crossentropy(y_true, y_pred) + dice_weight * dice_loss(y_true, y_pred)
    return bce_dice_loss_fixed


def bce_jaccardlog_loss(bce_weight=1.0, jaccardlog_weight=1.0):
    def bce_jaccardlog_loss_fixed(y_true, y_pred):
        return bce_weight * binary_crossentropy(y_true, y_pred) + jaccardlog_weight * jaccard_coef_logloss(y_true,
                                                                                                           y_pred)
    return bce_jaccardlog_loss_fixed


def bce_tversky_loss(bce_weight=1.0, tversky_weight=1.0):
    def bce_tversky_loss_fixed(y_true, y_pred):
        return bce_weight * binary_crossentropy(y_true, y_pred) + tversky_weight * tversky_loss(y_true,
                                                                                                y_pred)
    return bce_tversky_loss_fixed


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


def focal_dice_loss(focal_weight=1.0, dice_weight=1.0):
    def focal_dice_loss_fixed(y_true, y_pred):
        return focal_weight * focal_loss()(y_true, y_pred) + dice_weight * dice_coef_loss(y_true, y_pred)
    return focal_dice_loss_fixed


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
