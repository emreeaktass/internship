import tensorflow as tf
import Dataset as D
import DataLoader as DL
import os
import matplotlib.pyplot as plt
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

    path = "coco/val2014/dataset"
    X_train, X_test, y_train, y_test, X_val, y_val = D.get_splitted_datas(path)
    test_generator = DL.get_generator(X_test, y_test, 1)


    model = tf.keras.models.load_model('record1/', compile =False)

    for i in test_generator:
        y_pred = model.predict(i[0])
        plt.figure(1)
        plt.imshow(i[0][0], cmap='gray')
        plt.figure(2)
        plt.imshow(y_pred[0])
        plt.show()
