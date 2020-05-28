import random

import demjson
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow import keras
from tensorflow.keras import layers, backend
from tensorflow.keras.utils import plot_model

from notekeras.feature import ParseFeatureConfig

backend.set_floatx('float32')


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


def compare1():
    fields = [('age', 'int32'),
              ('trestbps', 'int32'),
              ('chol', 'int32'),
              ('thalach', 'int32'),
              ('oldpeak', 'int32'),
              ('slope', 'int32'),
              ('ca', 'int32'),
              ('thal', 'string'),
              ]

    # 将源数据的变量输入进来
    feature_dict = {}
    for field in fields:
        field_name = field[0]
        if field[1] == 'int32':
            field_type = tf.dtypes.int32
        else:
            field_type = tf.dtypes.string

        feature_dict[field_name] = tf.keras.Input((1,), dtype=field_type, name=field_name)

    # 对源数据字段进行处理
    feature_columns = []
    for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
        feature_columns.append(feature_column.numeric_column(header))

    age = feature_column.numeric_column("age")
    age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    feature_columns.append(age_buckets)

    thal = feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
    feature_columns.append(feature_column.indicator_column(thal))
    feature_columns.append(feature_column.embedding_column(thal, dimension=8))

    # crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
    # crossed_feature = feature_column.indicator_column(crossed_feature)
    # feature_columns.append(crossed_feature)

    feature_layer = layers.DenseFeatures(feature_columns)

    l0 = feature_layer(feature_dict)
    l1 = layers.Dense(128, activation='relu')(l0)
    l2 = layers.Dense(128, activation='relu')(l1)
    l3 = layers.Dense(1, activation='sigmoid')(l2)

    model = keras.models.Model(inputs=list(feature_dict.values()), outputs=[l3])
    model.compile(optimizer='adam', loss='binary_crossentropy', )
    model.summary()
    plot_model(model, to_file='feature.png', show_shapes=True)

    model.fit(train_ds, validation_data=val_ds, epochs=5)


def compare2():
    feature_json = open('model_feature.json', 'r').read()
    feature_json = demjson.decode(feature_json)
    parse = ParseFeatureConfig()

    l0 = parse.parse_feature_json(feature_json['layer0'])
    # l01 = parse.parse_feature_json(feature_json['layer4'])

    la1, la1_length = parse.parse_sequence_feature_json(feature_json['layer3'])
    la2, la2_length = parse.parse_sequence_feature_json(feature_json['layer2'])
    # la3, la3_length = parse.parse_sequence_feature_json(feature_json['layer5'])

    lb1 = tf.keras.backend.mean(la1, axis=1)
    lb2 = tf.keras.backend.mean(la2, axis=1)
    # lb3 = la3

    l0 = tf.keras.backend.concatenate([l0, lb1, lb2])

    l1 = layers.Dense(128, activation='relu')(l0)
    l2 = layers.Dense(64, activation='relu')(l1)
    l3 = layers.Dense(1, activation='sigmoid')(l2)

    model = keras.models.Model(inputs=list(parse.feature_dict.values()), outputs=[l3])
    model.compile(optimizer='adam', loss='binary_crossentropy', )
    model.summary()
    plot_model(model, to_file='feature2.png', show_shapes=True)

    model.fit(train_ds, validation_data=val_ds, epochs=5)


compare2()
