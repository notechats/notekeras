import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from tensorflow import keras
from tensorflow.keras import Model, activations
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Input, Layer
from tensorflow.keras.optimizers import Adam

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class FM(Layer):
    def __init__(self, output_dim, latent=10,  activation='relu', **kwargs):
        self.latent = latent
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.b = self.add_weight(name='W0',
                                 shape=(self.output_dim,),
                                 trainable=True,
                                 initializer='zeros')
        self.w = self.add_weight(name='W',
                                 shape=(input_shape[1], self.output_dim),
                                 trainable=True,
                                 initializer='random_uniform')
        self.v = self.add_weight(name='V',
                                 shape=(input_shape[1], self.latent),
                                 trainable=True,
                                 initializer='random_uniform')
        super(FM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        x_square = K.square(x)

        xv = K.square(K.dot(x, self.v))
        xw = K.dot(x, self.w)

        p = 0.5*K.sum(xv-K.dot(x_square, K.square(self.v)), 1)

        rp = K.repeat_elements(K.reshape(p, (-1, 1)), self.output_dim, axis=-1)

        f = xw + rp + self.b

        output = K.reshape(f, (-1, self.output_dim))

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.output_dim


data = load_breast_cancer()["data"]
target = load_breast_cancer()["target"]

K.clear_session()
print(target)
inputs = Input(shape=(30,))
out = FM(20)(inputs)
out = Dense(15, activation='sigmoid')(out)
out = Dense(1, activation='sigmoid')(out)

model = Model(inputs=inputs, outputs=out)
model.compile(loss='mse',
              optimizer='adam',
              metrics=['acc'])
model.summary()

h = model.fit(data, target, batch_size=1, epochs=10, validation_split=0.2)
