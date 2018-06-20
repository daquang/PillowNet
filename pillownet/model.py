from keras import backend as K
from keras.layers import Input, Conv1D, MaxPooling1D, concatenate
from keras.models import Model
from .layer import compose, Conv1D_BN_Leaky, Conv1DTranspose


def pillownet_body(seq_len, num_anchors, num_tasks, revcomp=False):
    input_dna = Input(shape=(seq_len, 4))
    x1 = compose(Conv1D_BN_Leaky(filters=16, kernel_size=25, padding='valid'),
                 MaxPooling1D(pool_size=2, strides=2),
                 Conv1D_BN_Leaky(filters=32, kernel_size=25, padding='valid'),
                 MaxPooling1D(pool_size=2, strides=2),
                 Conv1D_BN_Leaky(filters=64, kernel_size=25, padding='valid'),
                 MaxPooling1D(pool_size=2, strides=2),
                 Conv1D_BN_Leaky(filters=128, kernel_size=25, padding='valid'),
                 MaxPooling1D(pool_size=2, strides=2),
                 Conv1D_BN_Leaky(filters=256, kernel_size=25, padding='valid'),
                 MaxPooling1D(pool_size=2, strides=2),
                 Conv1D_BN_Leaky(filters=512, kernel_size=25, padding='valid'),
                 MaxPooling1D(pool_size=2, strides=1),
                 Conv1D_BN_Leaky(filters=1024, kernel_size=25, padding='valid'),
                 Conv1D_BN_Leaky(filters=256, kernel_size=1, padding='valid')
                 )(input_dna)
    y1 = compose(Conv1D_BN_Leaky(filters=512, kernel_size=25),
                 Conv1D(filters=num_anchors*num_tasks*3, kernel_size=1))(x1)

    model = Model([input_dna], [y1])
    return model


def localization():
    return None


def get_unet():
    size = 4096
    channel = 4
    inputs = Input((size, channel)) #4096
    print(inputs.shape)
    conv1 = Conv1D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv1D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1) #2048

    conv2 = Conv1D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv1D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2) #1024

    conv3 = Conv1D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv1D(128, 4, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3) #512

    conv4 = Conv1D(256, 3, activation='relu', padding='same')(pool3)#619
    conv4 = Conv1D(256, 4, activation='relu', padding='same')(conv4)#616
    pool4 = MaxPooling1D(pool_size=2)(conv4) #256

    conv5 = Conv1D(512, 3, activation='relu', padding='same')(pool4)#306
    conv5 = Conv1D(512, 3, activation='relu', padding='same')(conv5)#304

    up6 = concatenate([Conv1DTranspose(conv5,256, 2, strides=2, padding='same'), conv4], axis=2)
    #up6 = Conv1DTranspose(conv5,256, 2, strides=2, padding='same')
    conv6 = Conv1D(256, 4, activation='relu', padding='same')(up6)
    conv6 = Conv1D(256, 3, activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv1DTranspose(conv6,128, 2, strides=2, padding='same'), conv3], axis=2)
   # up7 = Conv1DTranspose(conv6,256, 2, strides=2, padding='same')
    conv7 = Conv1D(128, 4, activation='relu', padding='same')(up7)
    conv7 = Conv1D(128, 3, activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv1DTranspose(conv7,64, 2, strides=2, padding='same'), conv2], axis=2)
   # up8 = Conv1DTranspose(conv7, 64, 2, strides=2, padding='same')
    conv8 = Conv1D(64, 3, activation='relu', padding='same')(up8)
    conv8 = Conv1D(64, 3, activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv1DTranspose(conv8, 32, 2, strides=2, padding='same'), conv1], axis=2)
    #up9 = Conv1DTranspose(conv8,32, 2, strides=2, padding='same')
    conv9 = Conv1D(32, 3, activation='relu', padding='same')(up9)
    conv9 = Conv1D(32, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv1D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    #model.compile(optimizer=Adam(lr=1e-4,beta_1=0.9, beta_2=0.999,decay=0.0), loss=crossentropy_cut, metrics=[dice_coef]) ## you can change the crossentropy loss to dice loss

    return model