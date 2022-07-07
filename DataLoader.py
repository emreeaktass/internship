import random
import numpy as np

from tensorflow.keras.utils import Sequence


class DataLoader(Sequence):
    def __len__(self):

        return int(np.ceil(len(self.X) / self.batch_size))

    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.indices = list(range(len(self.X)))
        self.length = len(self.X)

        self.batch_size = batch_size
        random.shuffle(self.indices)

        self.xrange = 0
        if self.length % batch_size == 0:
            self.xrange = self.length // batch_size
        else:
            self.xrange = self.length // batch_size + 1

        self.batches_indices = [self.indices[a:]
                                if (self.length - a < self.batch_size) else self.indices[a:a + self.batch_size] for a in
                                range(0, self.length, self.batch_size)]

    def __getitem__(self, idx):

        self.batch_train = []
        self.batch_test = []

        for i in range(len(self.batches_indices[idx])):
            self.batch_train.append(np.load(self.X[self.batches_indices[idx][i]]))
            self.batch_test.append(np.load(self.y[self.batches_indices[idx][i]]))

        self.data = [i for i in self.batch_train]
        self.mask = [i for i in self.batch_test]
        self.data = np.array(self.data).astype(np.float32)
        self.mask = np.array(self.mask).astype(np.float32)
        self.data = self.data / 255.
        self.mask = self.mask / 255.
        self.data = self.data.reshape((len(self.data), 256, 256, 1))
        self.mask = self.mask.reshape((len(self.mask), 256, 256, 3))
        return self.data, self.mask


def get_generator(X, y, batch_size):
    return DataLoader(X, y, batch_size)
