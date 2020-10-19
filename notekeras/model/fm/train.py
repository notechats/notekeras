import os

import numpy as np
import tensorflow as tf
from notedata.dataset.datas import CriteoData
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam

from .model import AFM, FFM, FM, NFM
from .model2 import DeepFM
from .model3 import xDeepFM

criteo = CriteoData()


def download(mode=1):
    criteo.download(mode=mode)


def train_fm(mode=1):
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


def train_afm(mode=1):
    # ========================= Hyper Parameters =======================
    read_part = True
    sample_num = 100000
    test_size = 0.2

    embed_dim = 8

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10

    # ========================== Create dataset =======================
    feature_columns, train, test = criteo.build_dataset(mode=mode)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    mode = 'avg'  # 'max', 'avg'
    model = AFM(feature_columns, mode)
    model.summary()
    # ============================model checkpoint======================
    # check_path = 'save/afm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ===========================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(
            monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint,
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])


def train_ffm(mode=1):
    # ========================= Hyper Parameters =======================
    read_part = True
    sample_num = 100000
    test_size = 0.2

    k = 8

    learning_rate = 0.001
    batch_size = 512
    epochs = 5

    # ========================== Create dataset =======================
    feature_columns, train, test = criteo.build_dataset(mode=mode)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = FFM(feature_columns=feature_columns, k=k)
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


def train_nfm(mode=1):
    # ========================= Hyper Parameters =======================
    read_part = True
    sample_num = 5000000
    test_size = 0.2

    embed_dim = 8
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10
    # ========================== Create dataset =======================
    feature_columns, train, test = criteo.build_dataset(mode=mode)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = NFM(feature_columns, hidden_units, dnn_dropout=dnn_dropout)
    model.summary()
    # ============================model checkpoint======================
    # check_path = 'save/nfm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ===========================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(
            monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])


def train_deep_fm(mode=1, fm_type=1):
    # ========================= Hyper Parameters =======================
    # you can modify your file path

    read_part = True
    sample_num = 5000000
    test_size = 0.2

    embed_dim = 8
    k = 10
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10

    # ========================== Create dataset =======================
    feature_columns, train, test = criteo.build_dataset(mode=mode)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = DeepFM(feature_columns, k=k,
                   hidden_units=hidden_units, dnn_dropout=dnn_dropout, fm_type=fm_type)
    model.summary()
    # ============================model checkpoint======================
    # check_path = '../save/deepfm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
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
        callbacks=[EarlyStopping(
            monitor='val_loss', patience=1, restore_best_weights=True)],  # checkpoint,
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])


def train_x_deep_fm(mode=1):
    # ========================= Hyper Parameters =======================

    read_part = True
    sample_num = 500000
    test_size = 0.2

    embed_dim = 8
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]
    cin_size = [128, 128]

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10
    # ========================== Create dataset =======================
    feature_columns, train, test = criteo.build_dataset(mode=mode)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = xDeepFM(feature_columns, hidden_units, cin_size)
    model.summary()
    # ============================model checkpoint======================
    # check_path = 'save/xdeepfm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ===========================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(
            monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])
