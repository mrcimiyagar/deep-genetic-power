import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size=24, shuffle=False):
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        x = self.x[index * self.batch_size: (index + 1) * self.batch_size, ]
        y = self.y[index * self.batch_size * 24: (index + 1) * self.batch_size * 24, ]

        return x, y
