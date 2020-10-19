import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                     Embedding, Flatten, Input, Layer)
from tensorflow.keras.regularizers import l2


class FactorizationMachine(Layer):

    def __init__(self,
                 output_dim=1,
                 factor_dim=10,
                 name='FM',
                 # activation="relu",
                 activation=None,
                 use_weight=True,
                 use_bias=True,
                 kernal_reg=1e-4,
                 weight_reg=1e-4,
                 *args,
                 **kwargs):
        """
        Factorization Machines
        :param output_dim: 输出维度
        :param factor_dim: 隐含向量维度
        :param w_reg: the regularization coefficient of parameter w
        :param v_reg: the regularization coefficient of parameter v
        """
        super(FactorizationMachine, self).__init__(name=name, *args, **kwargs)
        self.output_dim = output_dim
        self.factor_dim = factor_dim
        self.activate = activation
        self.user_weight = use_weight
        self.use_bias = use_bias
        self.kernal_reg = kernal_reg
        self.weight_reg = weight_reg
        self.weight = self.bias = self.kernel = None
        self.activate_layer = None

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.factor_dim),
                                      initializer='glorot_uniform',
                                      regularizer=l2(self.kernal_reg),
                                      trainable=True)
        if self.user_weight:
            self.weight = self.add_weight(name='weight',
                                          shape=(
                                              input_shape[1], self.output_dim),
                                          initializer='glorot_uniform',
                                          regularizer=l2(self.weight_reg),
                                          trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.output_dim,),
                                        initializer='zeros',
                                        trainable=True)

        self.activate_layer = keras.activations.get(self.activate)
        super(FactorizationMachine, self).build(input_shape)

    def call(self, inputs, **kwargs):
        xv_a = K.square(K.dot(inputs, self.kernel))
        xv_b = K.dot(K.square(inputs), K.square(self.kernel))
        p = 0.5 * K.sum(xv_a - xv_b, 1)
        xv = K.repeat_elements(K.reshape(p, (-1, 1)), self.output_dim, axis=-1)

        res = xv
        if self.user_weight:
            res = res + K.dot(inputs, self.weight)
        if self.use_bias:
            res = res + self.bias

        output = K.reshape(res, (-1, self.output_dim))

        if self.activate_layer is not None:
            output = self.activate_layer(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.output_dim


FM = FactorizationMachine



class FFM(Layer):
    def __init__(self, dense_feature_columns, sparse_feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        """

        :param dense_feature_columns:
        :param sparse_feature_columns:
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
                :param v_reg: the regularization coefficient of parameter v
        """
        super(FFM, self).__init__()
        self.dense_feature_columns = dense_feature_columns
        self.sparse_feature_columns = sparse_feature_columns
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        self.feature_num = sum([feat['feat_num'] for feat in self.sparse_feature_columns]) \
            + len(self.dense_feature_columns)
        self.field_num = len(self.dense_feature_columns) + \
            len(self.sparse_feature_columns)

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_num, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.v = self.add_weight(name='v',
                                 shape=(self.feature_num,
                                        self.field_num, self.k),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        stack = dense_inputs
        # one-hot encoding
        for i in range(sparse_inputs.shape[1]):
            stack = tf.concat(
                [stack, tf.one_hot(sparse_inputs[:, i],
                                   depth=self.sparse_feature_columns[i]['feat_num'])], axis=-1)
        # first order
        first_order = self.w0 + tf.matmul(tf.concat(stack, axis=-1), self.w)
        # field second order
        second_order = 0
        field_f = tf.tensordot(stack, self.v, axes=[1, 0])
        for i in range(self.field_num):
            for j in range(i+1, self.field_num):
                second_order += tf.reduce_sum(
                    tf.multiply(field_f[:, i], field_f[:, j]),
                    axis=1, keepdims=True
                )

        return first_order + second_order
