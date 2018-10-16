import numpy as np
from .loss import dice_coef_loss, focal_loss, bce_dice_loss
from keras.layers import Input, Conv1D, MaxPooling1D, Dense, concatenate, Flatten, Average, Maximum, Cropping1D,\
    Bidirectional, LSTM, CuDNNLSTM, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from .layer import conv1dtranspose, conv1d_block, ReverseComplement, Reverse


def sliding(size=1018, input_channel=4, output_channel=1, filters=32, kernel_size=20, depth=1, pool_size=10,
            dense_units=200, recurrent=True):
    if isinstance(input_channel, int):
        if input_channel < 1:
            raise ValueError('input_channel must be positive integer or iterable of positive integers')
        inputs = Input((size, input_channel))
        x = inputs
        inputs = [inputs]
    else:
        try:
            inputs = [Input((size, i)) for i in input_channel]
            x = concatenate(inputs, axis=2)
        except TypeError:
            raise ValueError('input_channel must be positive integer or iterable of positive integers')

    for i in range(depth):
        x = conv1d_block(filters=filters * 2 ** i, kernel_size=kernel_size, padding='valid')(x)
        x = MaxPooling1D(pool_size=pool_size, strides=pool_size)(x)

    if recurrent:
        if len(K.tensorflow_backend._get_available_gpus()) == 0:
            recurrent_layer = Bidirectional(LSTM(units=filters * 2 ** depth, return_sequences=True))
        else:
            recurrent_layer = Bidirectional(CuDNNLSTM(units=filters * 2 ** depth, return_sequences=True))
        x = recurrent_layer(x)
    x = Flatten()(x)
    x = Dense(units=dense_units, activation='relu')(x)
    x = Dense(units=output_channel, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=[x])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def unet(size=3084, input_channel=4, output_channel=1, filters=32, kernel_size=11, depth=5, crop=True, skip=False,
         recurrent=False, loss='bce_dice'):
    if not isinstance(size, int) or size < 1:
        raise ValueError('size must be a positive integer')

    if not isinstance(output_channel, int) or output_channel < 1:
        raise ValueError('output_channel must be a positive integer')

    if not isinstance(filters, int) or filters < 1:
        raise ValueError('filters must be a positive integer')

    if not isinstance(kernel_size, int) or kernel_size < 1:
        raise ValueError('kernel_size must be a positive integer')

    if not isinstance(depth, int) or depth < 1:
        raise ValueError('depth must be a positive integer')

    padding = 'valid' if crop else 'same'

    if isinstance(input_channel, int):
        if input_channel < 1:
            raise ValueError('input_channel must be positive integer or iterable of positive integers')
        inputs = Input((size, input_channel))
        x = inputs
        inputs = [inputs]
    else:
        try:
            inputs = [Input((size, i)) for i in input_channel]
            x = concatenate(inputs, axis=2)
        except TypeError:
            raise ValueError('input_channel must be positive integer or iterable of positive integers')

    # left
    left_activations = []
    left_sizes = []
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
    if crop:
        x_size = x_size - kernel_size + 1 - kernel_size + 1
    if x_size < 1:  # sequence size only decreases if valid padding is used
        raise ValueError('size is too small. Consider increasing size or decreasing depth/kernel_size')
    middle_layer = conv1d_block(filters=filters * 2 ** depth, kernel_size=kernel_size, padding=padding)
    middle_activation = middle_layer(x)
    if recurrent:
        if len(K.tensorflow_backend._get_available_gpus()) == 0:
            recurrent_layer = Bidirectional(LSTM(units=filters * 2 ** depth, return_sequences=True))
        else:
            recurrent_layer = Bidirectional(CuDNNLSTM(units=filters * 2 ** depth, return_sequences=True))
        middle_activation = recurrent_layer(middle_activation)

    # right
    right_activations = []
    right_sizes = []
    x = middle_activation
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
                x = concat_activation
            else:
                if x_size != left_size:
                    raise ValueError('size and depth are incompatible. Make sure size is a multiple of 2^depth')
                x = concatenate([transpose_activation, left_activation], axis=2)
        else:
            x = transpose_activation
        right_layer = conv1d_block(filters=filters * 2 ** i, kernel_size=kernel_size, padding=padding)
        x = right_layer(x)
        if crop:
            x_size = x_size - kernel_size + 1 - kernel_size + 1
        right_activations.append(x)

    # final

    final_size = x_size
    conv_final = Conv1D(output_channel, 1, activation='sigmoid')(right_activations[-1])

    model = Model(inputs=inputs, outputs=[conv_final])

    if loss == 'bce':
        loss_func = 'binary_crossentropy'
    elif loss == 'dice':
        loss_func = dice_coef_loss
    elif loss == 'focal':
        loss_func = focal_loss()
    elif loss == 'bce_dice':
        loss_func = bce_dice_loss()
    else:
        raise ValueError('Loss must be one of the following: bce, dice, bce_dice')
    model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, decay=0.0), loss=loss_func,
                  metrics=['binary_accuracy', 'binary_crossentropy', dice_coef_loss])

    if final_size != model.output_shape[1]:
        print(final_size, model.output.shape[1].value)
        raise ValueError('Something went wrong in the size calculation')
    return model


def double_stranded(model, use_maximum=False):
    inputs = model.inputs
    inputs_rc = []
    for i in range(len(inputs)):
        if i == 0: # Only the first input is DNA, which must be reverse complemented, not reversed
            inputs_rc.append(ReverseComplement()(inputs[i]))
        else:
            inputs_rc.append(Reverse()(inputs[i]))
    output = model.outputs[0]
    output_rc = model(inputs_rc)
    if len(model.output_shape) == 3: # If the model is a u-net, the output must be reversed
        output_rc = Reverse()(output_rc)
    merge_layer = Maximum() if use_maximum else Average()
    outputs_merge = merge_layer([output, output_rc])
    model_merge = Model(inputs=inputs, outputs=outputs_merge)
    model_merge.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)
    return model_merge
