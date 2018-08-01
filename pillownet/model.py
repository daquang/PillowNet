import numpy as np
from .loss import dice_coef, crossentropy_cut
from keras.layers import Input, Conv1D, MaxPooling1D, Dense, concatenate, Flatten, Average, Maximum
from keras.models import Model, Sequential
from keras.optimizers import Adam
from .layer import compose, conv1d_batchnorm_leaky, conv1dtranspose, conv1d_block, ReverseComplement, Reverse


def pillownet_body(seq_len, num_anchors, num_tasks, revcomp=False):
    input_dna = Input(shape=(seq_len, 4))
    x1 = compose(conv1d_batchnorm_leaky(filters=16, kernel_size=25, padding='valid'),
                 MaxPooling1D(pool_size=2, strides=2),
                 conv1d_batchnorm_leaky(filters=32, kernel_size=25, padding='valid'),
                 MaxPooling1D(pool_size=2, strides=2),
                 conv1d_batchnorm_leaky(filters=64, kernel_size=25, padding='valid'),
                 MaxPooling1D(pool_size=2, strides=2),
                 conv1d_batchnorm_leaky(filters=128, kernel_size=25, padding='valid'),
                 MaxPooling1D(pool_size=2, strides=2),
                 conv1d_batchnorm_leaky(filters=256, kernel_size=25, padding='valid'),
                 MaxPooling1D(pool_size=2, strides=2),
                 conv1d_batchnorm_leaky(filters=512, kernel_size=25, padding='valid'),
                 MaxPooling1D(pool_size=2, strides=1),
                 conv1d_batchnorm_leaky(filters=1024, kernel_size=25, padding='valid'),
                 conv1d_batchnorm_leaky(filters=256, kernel_size=1, padding='valid')
                 )(input_dna)
    y1 = compose(conv1d_batchnorm_leaky(filters=512, kernel_size=25),
                 Conv1D(filters=num_anchors*num_tasks*3, kernel_size=1))(x1)

    model = Model([input_dna], [y1])
    return model


def sliding_model(seq_len=1024):
    size = seq_len
    channel = 4
    model = Sequential()
    model.add(Conv1D(filters=32,
                     kernel_size=26,
                     strides=1,
                     padding='valid',
                     activation='relu',
                     input_shape=(size, channel),
                     ))
    model.add(MaxPooling1D(pool_size=13,
                           strides=13))
    model.add(Flatten())
    model.add(Dense(units=975, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def localization():
    return None


def unet_model(size=1024, input_channel=4, output_channel=1, filters=32, kernel_size=7, depth=5):
    if type(depth) is not int or depth < 1:
        raise ValueError('depth must be a positive integer of at least 1')

    inputs = Input((size, input_channel))

    # left
    left_activations = []
    x = inputs
    for i in range(depth):
        left_layer = conv1d_block(filters=filters*2**i, kernel_size=kernel_size, padding='same')
        x = left_layer(x)
        left_activations.append(x)
        x = MaxPooling1D(pool_size=2)(x)

    # middle
    middle_layer = conv1d_block(filters=filters*2**depth, kernel_size=kernel_size, padding='same')
    middle_activation = middle_layer(x)

    # right
    right_activations = []
    x = middle_activation
    for i in range(depth-1, -1, -1):
        left_activation = left_activations[i]
        transpose_activation = conv1dtranspose(filters*2**i, 2, strides=2, padding='same')(x)
        concat_activation = concatenate([transpose_activation, left_activation], axis=2)
        right_layer = conv1d_block(filters=filters*2**i, kernel_size=kernel_size)
        x = right_layer(concat_activation)
        right_activations.append(x)

    # final

    conv_final = Conv1D(output_channel, 1, activation='sigmoid')(right_activations[-1])

    model = Model(inputs=[inputs], outputs=[conv_final])

    model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, decay=0.0), loss=crossentropy_cut, metrics=[dice_coef])

    return model


def unet_model_old(size=1024, input_channel=4, output_channel=1, filters=32, kernel_size=7):
    inputs = Input((size, input_channel))
    conv1 = Conv1D(filters*2**0, kernel_size, activation='relu', padding='same')(inputs)
    conv1 = Conv1D(filters*2**0, kernel_size, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(filters*2**1, kernel_size, activation='relu', padding='same')(pool1)
    conv2 = Conv1D(filters*2**1, kernel_size, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(filters*2**2, kernel_size, activation='relu', padding='same')(pool2)
    conv3 = Conv1D(filters*2**2, kernel_size, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(filters*2**3, kernel_size, activation='relu', padding='same')(pool3)
    conv4 = Conv1D(filters*2**3, kernel_size, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling1D(pool_size=2)(conv4)

    conv5 = Conv1D(filters*2**4, kernel_size, activation='relu', padding='same')(pool4)
    conv5 = Conv1D(filters*2**4, kernel_size, activation='relu', padding='same')(conv5)
    pool5 = MaxPooling1D(pool_size=2)(conv5)

    conv6 = Conv1D(filters*2**5, kernel_size, activation='relu', padding='same')(pool5)
    conv6 = Conv1D(filters*2**5, kernel_size, activation='relu', padding='same')(conv6)

    up7 = concatenate([conv1dtranspose(filters*2**4, kernel_size=2, strides=2, padding='same')(conv6), conv5], axis=2)
    conv7 = Conv1D(filters*2**4, kernel_size, activation='relu', padding='same')(up7)
    conv7 = Conv1D(filters*2**4, kernel_size, activation='relu', padding='same')(conv7)

    up8 = concatenate([conv1dtranspose(filters*2**3, kernel_size=2, strides=2, padding='same')(conv7), conv4], axis=2)
    conv8 = Conv1D(filters*2**3, kernel_size, activation='relu', padding='same')(up8)
    conv8 = Conv1D(filters*2**3, kernel_size, activation='relu', padding='same')(conv8)

    up9 = concatenate([conv1dtranspose(filters*2**2, kernel_size=2, strides=2, padding='same')(conv8), conv3], axis=2)
    conv9 = Conv1D(filters*2**2, kernel_size, activation='relu', padding='same')(up9)
    conv9 = Conv1D(filters*2**2, kernel_size, activation='relu', padding='same')(conv9)

    up10 = concatenate([conv1dtranspose(filters*2**1, kernel_size=2, strides=2, padding='same')(conv9), conv2], axis=2)
    conv10 = Conv1D(filters*2**1, kernel_size, activation='relu', padding='same')(up10)
    conv10 = Conv1D(filters*2**1, kernel_size, activation='relu', padding='same')(conv10)

    up11 = concatenate([conv1dtranspose(filters*2**0, kernel_size=2, strides=2, padding='same')(conv10), conv1], axis=2)
    conv11 = Conv1D(filters*2**0, kernel_size, activation='relu', padding='same')(up11)
    conv11 = Conv1D(filters*2**0, kernel_size, activation='relu', padding='same')(conv11)

    conv12 = Conv1D(output_channel, 1, activation='sigmoid')(conv11)

    model = Model(inputs=[inputs], outputs=[conv12])

    model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, decay=0.0), loss=crossentropy_cut, metrics=[dice_coef])

    return model


def double_stranded_model(model, use_maximum=False):
    inputs = model.inputs
    inputs_rc = [ReverseComplement()(i) for i in inputs]
    output = model.outputs[0]
    output_rc = Reverse()(model(inputs_rc))  # If the underlying model is a u-net, the output must be reversed
    if use_maximum:
        outputs_merge = Maximum()([output, output_rc])
    else:
        outputs_merge = Average()([output, output_rc])
    model_merge = Model(inputs=inputs, outputs=outputs_merge)
    model_merge.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)
    return model_merge
