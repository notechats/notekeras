import os

import numpy as np
import tensorflow as tf
from notedata.dataset.datas import CriteoData
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam

from .model import FM

criteo = CriteoData()


def download(model=1):
    criteo.download(mode=mode)


def train(mode=1):
    # ========================= Hyper Parameters =======================
    read_part = True
    sample_num = 100000
    test_size = 0.2

    k = 10

    learning_rate = 0.001
    batch_size = 512
    epochs = 5

    feature_columns, train, test = criteo.build_dataset(mode=mode)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = FM(feature_columns=feature_columns, k=k)
    model.summary()
    # ============================model checkpoint======================
    # check_path = '../save/fm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # ============================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ==============================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        # callbacks=[checkpoint],
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])
