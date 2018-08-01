from functools import reduce
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Conv1D, Conv2DTranspose, Lambda, Add, UpSampling1D, Concatenate, MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from .motif import load_meme


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


def conv1d_batchnorm_leaky(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, alpha=0.1):
    """Convolution1D followed by BatchNormalization and LeakyReLU."""
    return compose(
        Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate),
        BatchNormalization(),
        LeakyReLU(alpha=alpha))


def conv1dtranspose(filters, kernel_size, strides=2, padding='same'):
    return compose(
        Lambda(lambda x: K.expand_dims(x, axis=2)),
        Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding),
        Lambda(lambda x: K.squeeze(x, axis=2)))


def conv1d_block(filters=32, kernel_size=7, activation='relu', padding='same', depth=2):
    block = compose(*[Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)
                      for _ in range(depth)])
    return block


def convmaxpool1d_block(filters=32, kernel_size=7, activation='relu', padding='same', depth=2, pool_size=2):
    block = compose(*[Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)
                      for _ in range(depth)], MaxPooling1D(pool_size=pool_size))
    return block


class ReverseComplement(Layer):
    def call(self, x):
        return x[:, ::-1, ::-1]


class Reverse(Layer):
    def call(self, x):
        return x[:, ::-1, :]

