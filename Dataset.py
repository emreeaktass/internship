import os
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import Sequence

class Dataset(Sequence):
    def __init__(self, root):
        self.root = root

        self.data_file_names = list(sorted(os.listdir(os.path.join(root, "gray"))))
        self.mask_file_names = list(sorted(os.listdir(os.path.join(root, "rgb"))))

    def __getitem__(self, idx):
        self.data_name = os.path.join(self.root, "gray", self.data_file_names[idx])
        self.mask_name = os.path.join(self.root, "rgb", self.mask_file_names[idx])
        return self.data_name, self.mask_name

    def __len__(self):
        return len(self.data_file_names)


def get_splitted_datas(path):
    root = path
    dataset = Dataset(root)
    datas = [i[0] for i in dataset]
    masks = [i[1] for i in dataset]
    X_train, X_test, y_train, y_test = train_test_split(datas, masks, test_size=0.30, random_state=47)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.7, random_state=47)
    return X_train, X_test, y_train, y_test, X_val, y_val
