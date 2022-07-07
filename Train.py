import tensorflow as tf
from sklearn.model_selection import train_test_split
import Dataset as D
import DataLoader as DL
import Model2 as M
import os
import numpy as np
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            print("test gpu")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    #     try:
    #         tf.config.experimental.set_virtual_device_configuration(
    #             gpus[0],
    #             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
    #     except:
    #         pass

    path = "coco/val2014/dataset"
    X_train, X_test, y_train, y_test, X_val, y_val = D.get_splitted_datas(path)
    val_generator = DL.get_generator(X_val, y_val, 8)
    test_generator = DL.get_generator(X_test, y_test, 1)
    train_generator = DL.get_generator(X_train[:10000], y_train[:10000], 8)




    early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', min_delta=0, patience=3, verbose=0,
                    mode='auto', baseline=None, restore_best_weights=True
    )

    model = M.auto_encoder()
    print(model.summary())
    model.fit(train_generator, validation_data=val_generator, epochs=25, verbose=1, shuffle=False, callbacks=[early_stop])
    model.save('record2/')