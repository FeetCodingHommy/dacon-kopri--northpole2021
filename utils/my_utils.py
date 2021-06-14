import tensorflow as tf
import numpy as np
import pandas as pd

from utils.dacon_functions import train_map_func


# TO-DO:
# batch size 1 이상 입력 시 해당 batch size로 고정되어 길이가 다를 때 오류가 나는 문제 해결
BATCH_SIZE = 1


class DataGenerator(tf.keras.utils.Sequence):
    """

    출처:
    https://towardsdatascience.com/implementing-custom-data-generators-in-keras-de56f013581c
    https://cyc1am3n.github.io/2018/09/13/how-to-use-dataset-in-tensorflow.html
    """

    def __init__(
        self, 
        x_dirs, 
        y_dirs, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.df = pd.DataFrame.from_dict(
                {'X_dir': x_dirs, 'y_dir': y_dirs}
            )
        self.indices = self.df.index.tolist()
        self.on_epoch_end()

    def on_epoch_end(self):
        self.index = self.df.index.tolist() # np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __len__(self):
        # Denotes the number of batches per epoch
        # return floor(len(self.indices) / self.batch_size)
        return len(self.indices) // self.batch_size

    def __get_data(self, batch):
        # X.shape : (batch_size, *dim)
        # We can have multiple Xs and can return them as a list
        
        Xs, ys = list(), list()
        # Generate data
        for idx in batch:
            X, y = train_map_func(self.df["X_dir"][idx], self.df["y_dir"][idx])
            Xs.append(X)
            ys.append(y)
        
        return np.array(Xs), np.array(ys)

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indices of the batch
        index = self.index[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        batch = [self.indices[k] for k in index]

        # Generate data
        X, y = self.__get_data(batch)

        return X, y


def my_cycle(epoch):
    eps = 0.01      # 바닥
    ceil = 1 - eps  # 천장
    if epoch < 40:
        return (0.5 * np.cos((epoch/20) * 2*np.pi * epoch / 10) + 0.5) * ceil * (0.55 ** (epoch/10)) + eps
    else:
        return (0.5 * np.cos(8 * np.pi * epoch / 10) + 0.5) * ceil * (0.55 ** 4) + eps


def my_cycle_scheduler(epoch, lr):
    # 0~40: 1~0.01배 / 50~: 0.1~0.01배
    return lr * my_cycle(epoch)
