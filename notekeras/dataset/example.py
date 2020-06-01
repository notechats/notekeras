import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def get_data():
    def arr_c(n):
        res = []
        list = ["a", 'b', 'c', 'e', 'f']
        for i in range(0, n):
            random.shuffle(list)
            res.append(np.array(list[:4]))
        return res

    def arr_c2(n):
        res = []
        list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for i in range(0, n):
            random.shuffle(list)
            res.append(np.array(list[:4]))
        return res

    URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
    dataframe = pd.read_csv(URL)
    dataframe['arr'] = arr_c(len(dataframe))
    dataframe['arr2'] = arr_c(len(dataframe))
    dataframe['arr3'] = arr_c2(len(dataframe))
    dataframe['thal2'] = dataframe['thal']
    dataframe = pd.DataFrame(dataframe)

    train, test = train_test_split(dataframe, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        labels = dataframe.pop('target')
        # arr = np.vstack(dataframe.pop('arr'))
        data = dict(dataframe)
        # data['arr'] = arr

        ds = tf.data.Dataset.from_tensor_slices((data, labels))

        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds

    batch_size = 5
    train_d = df_to_dataset(train, batch_size=batch_size)
    val_d = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_d = df_to_dataset(test, shuffle=False, batch_size=batch_size)
    return train_d, val_d, test_d


train_ds, val_ds, test_ds = get_data()
