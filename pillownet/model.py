import numpy as np
from .loss import dice_coef, binary_crossentropy_cut, binary_accuracy_cut
from keras.layers import Input, Conv1D, MaxPooling1D, Dense, concatenate, Flatten, Average, Maximum, Cropping1D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from .layer import conv1dtranspose, conv1d_block, ReverseComplement, Reverse


def sliding(size=1024, input_channel=4, output_channel=1, filters=32, kernel_size=26, dense_units=975):
    model = Sequential()
    model.add(Conv1D(filters=filters,
                     kernel_size=kernel_size,
                     strides=1,
                     padding='valid',
                     activation='relu',
                     input_shape=(size, input_channel),
                     ))
    model.add(MaxPooling1D(pool_size=13,
                           strides=13))
    model.add(Flatten())
    model.add(Dense(units=dense_units, activation='relu'))
    model.add(Dense(units=output_channel, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


def unet(size=1024, input_channel=4, output_channel=1, filters=32, kernel_size=7, depth=5, crop=False, skip=True,
         use_dice=False):
    if type(size) is not int or size < 1:
        raise ValueError('size must be a positive integer')

    if type(input_channel) is not int or input_channel < 1:
        raise ValueError('input_channel must be a positive integer')

    if type(output_channel) is not int or output_channel < 1:
        raise ValueError('output_channel must be a positive integer')

    if type(filters) is not int or filters < 1:
        raise ValueError('filters must be a positive integer')

    if type(kernel_size) is not int or kernel_size < 1:
        raise ValueError('kernel_size must be a positive integer')

    if type(depth) is not int or depth < 1:
        raise ValueError('depth must be a positive integer')

    padding = 'valid' if crop else 'same'

    inputs = Input((size, input_channel))

    # left
    left_activations = []
    left_sizes = []
    x = inputs
    x_size = size
    for i in range(depth):
        left_layer = conv1d_block(filters=filters * 2 ** i, kernel_size=kernel_size, padding=padding)
        if crop:
            x_size = x_size - kernel_size + 1 - kernel_size + 1
            if x_size < 1:  # sequence size only decreases if valid padding is used
                raise ValueError('size is too small. Consider increasing size or decreasing depth/kernel_size')
        x = left_layer(x)
        left_activations.append(x)
        left_sizes.append(x_size)
        x_size //= 2
        x = MaxPooling1D(pool_size=2)(x)

    # middle
    middle_layer = conv1d_block(filters=filters * 2 ** depth, kernel_size=kernel_size, padding=padding)
    middle_activation = middle_layer(x)

    # right
    right_activations = []
    right_sizes = []
    x = middle_activation
    if crop:
        x_size = x_size - kernel_size + 1 - kernel_size + 1
    for i in range(depth - 1, -1, -1):
        left_activation = left_activations[i]
        left_size = left_sizes[i]
        transpose_layer = conv1dtranspose(filters * 2 ** i, 2, strides=2, padding='same')
        x_size *= 2
        transpose_activation = transpose_layer(x)
        right_sizes.append(x_size)
        if skip:
            if crop:
                left_crop_size = int(np.floor((left_size - x_size) / 2))
                right_crop_size = int(np.ceil((left_size - x_size) / 2))
                crop_activation = Cropping1D((left_crop_size, right_crop_size))(left_activation)
                concat_activation = concatenate([transpose_activation, crop_activation], axis=2)
                x_size = x_size - kernel_size + 1 - kernel_size + 1
                x = concat_activation
            else: # maybe just delete this part. this only happens if you don't have a power of 2
                if x_size != left_size:
                    raise ValueError('size and depth are incompatible. Make sure size is a multiple of 2^depth')
                x = concatenate([transpose_activation, left_activation], axis=2)
        else:
            x = transpose_activation
        right_layer = conv1d_block(filters=filters * 2 ** i, kernel_size=kernel_size, padding=padding)
        x = right_layer(x)
        right_activations.append(x)

    # final

    final_size = x_size
    conv_final = Conv1D(output_channel, 1, activation='sigmoid')(right_activations[-1])

    model = Model(inputs=[inputs], outputs=[conv_final])

    loss_func = dice_coef if use_dice else binary_crossentropy_cut
    model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, decay=0.0), loss=loss_func,
                  metrics=[binary_accuracy_cut, loss_func])

    return model


def double_stranded_model(model, use_maximum=False):
    inputs = model.inputs
    inputs_rc = [ReverseComplement()(i) for i in inputs]
    output = model.outputs[0]
    output_rc = Reverse()(model(inputs_rc))  # If the underlying model is a u-net, the output must be reversed
    merge_layer = Maximum() if use_maximum else Average()
    outputs_merge = merge_layer([output, output_rc])
    model_merge = Model(inputs=inputs, outputs=outputs_merge)
    model_merge.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)
    return model_merge
