import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dropout, UpSampling2D, concatenate
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


def auto_encoder(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="encoder_1")(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2), name="pool_1")(conv1)
    conv2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="encoder_2")(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), name="pool_2")(conv2)
    conv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="encoder_3")(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), name="pool_3")(conv3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="encoder_4")(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2), name="pool_4")(conv4)

    up6 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal', name="decoder_1_a")(
        UpSampling2D(size=(2, 2), name="up_1")(pool4))
    conv6 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="decoder_1_b")(up6)
    x1 = Conv2D(3, 1, activation='sigmoid', name="output_1")(conv6)

    up7 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal', name="decoder_2_a")(
        UpSampling2D(size=(2, 2), name="up_2")(conv6))
    conv7 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="decoder_2_b")(up7)

    up8 = Conv2D(8, 2, activation='relu', padding='same', kernel_initializer='he_normal', name="decoder_3_a")(
        UpSampling2D(size=(2, 2), name="up_3")(conv7))
    conv8 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="decoder_3_b")(up8)
    x2 = Conv2D(3, 1, activation='sigmoid', name="output_2")(conv8)

    up9 = Conv2D(4, 2, activation='relu', padding='same', kernel_initializer='he_normal', name="decoder_4_a")(
        UpSampling2D(size=(2, 2), name="up_4")(conv8))
    conv9 = Conv2D(4, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="decoder_4_b")(up9)
    conv10 = Conv2D(3, 1, activation='sigmoid', name="output_3")(conv9)

    model = Model(inputs=[inputs], outputs=[x1, x2, conv10])

    model.compile(optimizer=Adam(learning_rate=0.001), loss={"output_1" : my_loss, "output_2" : my_loss, "output_3" : my_loss,}, metrics=[my_coef])
    return model


def my_loss(y_true, y_pred):
    x1_ = tf.image.resize(y_true, [y_pred.shape[1], y_pred.shape[1]])
    # x2_ = tf.image.resize(y_true, [128, 128])

    first = tf.reduce_sum((x1_ - y_pred) * (x1_ - y_pred))
    # second = tf.reduce_sum((x2_ - y_pred[1]) * (x2_ - y_pred[1]))
    # third = tf.reduce_sum((y_true - y_pred[2]) * (y_true - y_pred[2]))

    fidelity = first

    return fidelity


def my_coef(y_true, y_pred):
    return 1 - my_loss(y_true, y_pred)


model = auto_encoder()
dot_img_file = 'model_2.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

print(model.summary())