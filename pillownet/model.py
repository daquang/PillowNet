import numpy as np
from .loss import dice_loss, focal_loss, bce_dice_loss, focal_dice_loss, tversky_loss, jaccard_coef_logloss,\
    bce_jaccardlog_loss, bce_tversky_loss
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


def simple_window(size=6700, input_channel=1, output_channel=1, kernel_size=1881, loss='mse_regression'):
    if not isinstance(size, int) or size < 1:
        raise ValueError('size must be a positive integer')

    if not isinstance(input_channel, int) or size < 1:
        raise ValueError('input_channel must be a positive integer')

    if not isinstance(output_channel, int) or output_channel < 1:
        raise ValueError('output_channel must be a positive integer')

    if not isinstance(kernel_size, int) or kernel_size < 1:
        raise ValueError('kernel_size must be a positive integer')

    if loss == 'mse_regression':
        loss_func = 'mse'
        output_activation = 'relu'
    elif loss == 'msle_regression':
        loss_func = 'msle'
        output_activation = 'relu'
    elif loss == 'poisson_regression':
        loss_func = 'poisson'
        output_activation = 'relu'
    else:
        raise ValueError('Loss must be one of the following: mse_regression, msle_regression, poisson_regression')
    metrics = ['mse', 'msle', 'poisson']

    inputs = Input((size, input_channel))
    conv_layer = Conv1D(filters=output_channel, kernel_size=kernel_size, activation=output_activation)
    outputs = conv_layer(inputs)
    model = Model(inputs=inputs, outputs=[outputs])
    model.compile(optimizer=Adam(lr=1e-5), loss=loss_func,
                  metrics=metrics)
    return model


def unet(size=6700, input_channel=4, output_channel=1, filters=32, kernel_size=11, depth=5, crop=True, skip=True,
         recurrent=False, motifs_layer=None, use_batchnorm=True, loss='bce_dice'):
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

    output_activation = 'sigmoid'
    if loss == 'bce':
        loss_func = 'binary_crossentropy'
        output_activation = 'sigmoid'
        metrics = ['binary_accuracy', 'binary_crossentropy', dice_loss]
    elif loss == 'dice':
        loss_func = dice_loss
        output_activation = 'sigmoid'
        metrics = ['binary_accuracy', 'binary_crossentropy', dice_loss]
    elif loss == 'focal':
        loss_func = focal_loss()
        output_activation = 'sigmoid'
        metrics = ['binary_accuracy', 'binary_crossentropy', dice_loss]
    elif loss == 'bce_dice':
        loss_func = bce_dice_loss()
        output_activation = 'sigmoid'
        metrics = ['binary_accuracy', 'binary_crossentropy', dice_loss]
    elif loss == 'focal_dice':
        loss_func = focal_dice_loss()
        metrics = ['binary_accuracy', 'binary_crossentropy', dice_loss]
    elif loss == 'tversky':
        loss_func = tversky_loss
        metrics = ['binary_accuracy', 'binary_crossentropy', dice_loss]
    elif loss == 'jaccardlog':
        loss_func = jaccard_coef_logloss
        metrics = ['binary_accuracy', 'binary_crossentropy', dice_loss]
    elif loss == 'bce_jaccardlog':
        loss_func = bce_jaccardlog_loss()
        metrics = ['binary_accuracy', 'binary_crossentropy', dice_loss]
    elif loss == 'bce_tversky':
        loss_func = bce_tversky_loss()
        metrics = ['binary_accuracy', 'binary_crossentropy', dice_loss]
    elif loss == 'mse_regression':
        loss_func = 'mse'
        output_activation = 'relu'
        metrics = ['mse', 'msle', 'poisson']
    elif loss == 'msle_regression':
        loss_func = 'msle'
        output_activation = 'relu'
        metrics = ['mse', 'msle', 'poisson']
    elif loss == 'poisson_regression':
        loss_func = 'poisson'
        output_activation = 'relu'
        metrics = ['mse', 'msle', 'poisson']
    else:
        raise ValueError('Invalid loss choice')

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

    if motifs_layer:
        if not (input_channel == 4 or input_channel[0] == 4):
            raise ValueError('You can only use a motifs layer if DNA signal is included')
        motifs_acts = motifs_layer(inputs[0])
        x = concatenate([x, motifs_acts], axis=2)

    # Apply batch normalization
    if use_batchnorm:
        x = BatchNormalization()(x)
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
    conv_final = Conv1D(output_channel, 1, activation=output_activation)(right_activations[-1])

    model = Model(inputs=inputs, outputs=[conv_final])

    model.compile(optimizer=Adam(lr=1e-5), loss=loss_func,
                  metrics=metrics)

    if final_size != model.output_shape[1]:
        print(final_size, model.output.shape[1].value)
        raise ValueError('Something went wrong in the size calculation')

    if not crop and final_size != size:
        raise ValueError('You specified a non-crop U-Net, but the input and output sizes do not match.')
    return model


def double_stranded(model, use_maximum=False):
    inputs = model.inputs
    inputs_rc = [ReverseComplement()(i) for i in inputs]
    output = model.outputs[0]
    output_rc = model(inputs_rc)
    if len(model.output_shape) == 3:  # If the model is a u-net, the output must be reversed
        output_rc = Reverse()(output_rc)
    merge_layer = Maximum() if use_maximum else Average()
    outputs_merge = merge_layer([output, output_rc])
    model_merge = Model(inputs=inputs, outputs=outputs_merge)
    model_merge.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)
    return model_merge
