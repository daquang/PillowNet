from functools import reduce
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Conv1D, Conv2DTranspose, Lambda, Add, UpSampling1D, Concatenate, MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU
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


def Conv1D_BN_Leaky(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, alpha=0.1):
    """Convolution1D followed by BatchNormalization and LeakyReLU."""
    return compose(
        Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate),
        BatchNormalization(),
        LeakyReLU(alpha=alpha))


def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x


class ReverseComplement(Layer):
    def call(self, x):
        return x[:, ::-1, ::-1]
