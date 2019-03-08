from functools import reduce
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Conv1D, Conv2DTranspose, Lambda, Add, UpSampling1D, Concatenate, MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization


def compose(*funcs):
    """
    Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    Adapted from: https://github.com/qqwweee/keras-yolo3
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def conv1d_batchnorm_leaky(filters=32, kernel_size=11, strides=1, padding='valid', dilation_rate=1):
    """Convolution1D followed by BatchNormalization and PReLU."""
    return compose(
        Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate),
        BatchNormalization(),
        PReLU())


def conv1d_leaky(filters=32, kernel_size=11, strides=1, padding='valid', dilation_rate=1):
    """Convolution1D followed by BatchNormalization and PReLU."""
    return compose(
        Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate),
        PReLU())


def conv1dtranspose(filters=32, kernel_size=11, strides=2, padding='same'):
    return compose(
        Lambda(lambda x: K.expand_dims(x, axis=2)),
        Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding),
        Lambda(lambda x: K.squeeze(x, axis=2)))


def conv1d_block(filters=32, kernel_size=11, activation='relu', padding='same', depth=2):
    block = compose(*[Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)
                      for _ in range(depth)])
    return block


def convmaxpool1d_block(filters=32, kernel_size=11, activation='relu', padding='same', depth=2, pool_size=2):
    block = compose(*[Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)
                      for _ in range(depth)], MaxPooling1D(pool_size=pool_size))
    return block


class Motifs(Conv1D):
    def __init__(self, ppms, smooth=1e-6):
        ppms_lens = [len(ppm) for ppm in ppms]
        max_ppms_lens = max(ppms_lens)
        pwm_weights = np.zeros((max_ppms_lens, 4, len(ppms)))
        for i in range(len(ppms)):
            ppm = ppms[i]
            pwm = ppm.copy()
            pwm[pwm < smooth] = smooth
            pwm = pwm / 0.25
            pwm = np.log2(pwm)
            pwm_weights[:len(pwm), :, i] = pwm[::-1]
        super().__init__(filters=len(ppms), strides=1, kernel_size=max_ppms_lens, padding='same',
                         weights=[pwm_weights], use_bias=False, trainable=False)


class ReverseComplement(Layer):
    def call(self, x):
        return x[:, ::-1, ::-1]


class Reverse(Layer):
    def call(self, x):
        return x[:, ::-1, :]

