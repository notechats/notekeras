import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                     Embedding, Input, Layer)
from tensorflow.keras.regularizers import l2


class FM_Layer(Layer):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        """
        Factorization Machines
        :param feature_columns: a list containing dense and sparse column feature information
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
        :param v_reg: the regularization coefficient of parameter v
        """
        super(FM_Layer, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.feature_length = sum([feat['feat_num'] for feat in self.sparse_feature_columns]) \
            + len(self.dense_feature_columns)
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.V = self.add_weight(name='V', shape=(self.k, self.feature_length),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        # one-hot encoding
        sparse_inputs = tf.concat(
            [tf.one_hot(sparse_inputs[:, i],
                        depth=self.sparse_feature_columns[i]['feat_num'])
             for i in range(sparse_inputs.shape[1])
             ], axis=1)
        stack = tf.concat([dense_inputs, sparse_inputs], axis=1)
        # first order
        first_order = self.w0 + tf.matmul(stack, self.w)
        # second order
        second_order = 0.5 * tf.reduce_sum(
            tf.pow(tf.matmul(stack, tf.transpose(self.V)), 2) -
            tf.matmul(tf.pow(stack, 2), tf.pow(tf.transpose(self.V), 2)), axis=1, keepdims=True)
        outputs = first_order + second_order
        return outputs


class FM(keras.Model):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        """
        Factorization Machines
        :param feature_columns: a list containing dense and sparse column feature information
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
                :param v_reg: the regularization coefficient of parameter v
        """
        super(FM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.fm = FM_Layer(feature_columns, k, w_reg, v_reg)

    def call(self, inputs, **kwargs):
        fm_outputs = self.fm(inputs)
        outputs = tf.nn.sigmoid(fm_outputs)
        return outputs

    def summary(self, **kwargs):
        dense_inputs = tf.keras.Input(
            shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = tf.keras.Input(
            shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call(
            [dense_inputs, sparse_inputs])).summary()


class AFM(keras.Model):
    def __init__(self, feature_columns, mode, activation='relu', embed_reg=1e-4):
        """
        AFM 
        :param feature_columns: A list. dense_feature_columns and sparse_feature_columns
        :param mode:A string. 'max'(MAX Pooling) or 'avg'(Average Pooling) or 'att'(Attention)
        :param activation: A string. Activation function of attention.
        :param embed_reg: A scalar. the regularizer of embedding
        """
        super(AFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.mode = mode
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        if self.mode == 'att':
            t = (len(self.embed_layers) - 1) * len(self.embed_layers) // 2
            self.attention_W = Dense(units=t, activation=activation)
            self.attention_dense = Dense(units=1, activation=None)

        self.dense = Dense(units=1, activation=None)

    def call(self, inputs):
        # Input Layer
        dense_inputs, sparse_inputs = inputs
        # Embedding Layer
        embed = [self.embed_layers['embed_{}'.format(i)](
            sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        # (None, len(sparse_inputs), embed_dim)
        embed = tf.transpose(tf.convert_to_tensor(embed), perm=[1, 0, 2])
        # Pair-wise Interaction Layer
        # for loop is badly
        element_wise_product_list = []
        # t = (len - 1) * len /2, k = embed_dim
        for i in range(embed.shape[1]):
            for j in range(i+1, embed.shape[1]):
                element_wise_product_list.append(
                    tf.multiply(embed[:, i], embed[:, j]))
        element_wise_product = tf.transpose(
            tf.stack(element_wise_product_list), [1, 0, 2])  # (None, t, k)
        # mode
        if self.mode == 'max':
            # MaxPooling Layer
            x = tf.reduce_sum(element_wise_product, axis=1)   # (None, k)
        elif self.mode == 'avg':
            # AvgPooling Layer
            x = tf.reduce_mean(element_wise_product, axis=1)  # (None, k)
        else:
            # Attention Layer
            x = self.attention(element_wise_product)  # (None, k)
        # Output Layer
        outputs = tf.nn.sigmoid(self.dense(x))

        return outputs

    def summary(self):
        dense_inputs = Input(
            shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(
            shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call(
            [dense_inputs, sparse_inputs])).summary()

    def attention(self, keys):
        a = self.attention_W(keys)  # (None, t, t)
        a = self.attention_dense(a)  # (None, t, 1)
        a_score = tf.nn.softmax(a)  # (None, t, 1)
        a_score = tf.transpose(a_score, [0, 2, 1])  # (None, 1, t)
        outputs = tf.reshape(tf.matmul(a_score, keys),
                             shape=(-1, keys.shape[2]))  # (None, k)
        return outputs


class FFM_Layer(Layer):
    def __init__(self, dense_feature_columns, sparse_feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        """

        :param dense_feature_columns:
        :param sparse_feature_columns:
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
                :param v_reg: the regularization coefficient of parameter v
        """
        super(FFM_Layer, self).__init__()
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


class FFM(tf.keras.Model):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        """
        FFM architecture
        :param feature_columns:  a list containing dense and sparse column feature information
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
                :param field_reg_reg: the regularization coefficient of parameter v
        """
        super(FFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.ffm = FFM_Layer(self.dense_feature_columns, self.sparse_feature_columns,
                             k, w_reg, v_reg)

    def call(self, inputs, **kwargs):
        result_ffm = self.ffm(inputs)
        outputs = tf.nn.sigmoid(result_ffm)

        return outputs

    def summary(self, **kwargs):
        dense_inputs = Input(
            shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(
            shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs],
                       outputs=self.call([dense_inputs, sparse_inputs])).summary()


class DNN(Layer):
    """
        Deep Neural Network
        """

    def __init__(self, hidden_units, activation='relu', dropout=0.):
        """
                :param hidden_units: A list. Neural network hidden units.
                :param activation: A string. Activation function of dnn.
                :param dropout: A scalar. Dropout number.
                """
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation)
                            for unit in hidden_units]
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x


class NFM(keras.Model):
    def __init__(self, feature_columns, hidden_units, dnn_dropout=0., activation='relu', bn_use=True, embed_reg=1e-4):
        """
        NFM architecture
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param hidden_units: A list. Neural network hidden units.
        :param activation: A string. Activation function of dnn.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param bn_use: A Boolean. Use BatchNormalization or not.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(NFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.bn = BatchNormalization()
        self.bn_use = bn_use
        self.dnn_network = DNN(hidden_units, activation, dnn_dropout)
        self.dense = Dense(1)

    def call(self, inputs):
        # Inputs layer
        dense_inputs, sparse_inputs = inputs
        # Embedding layer
        embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                 for i in range(sparse_inputs.shape[1])]
        # (None, filed_num, embed_dim)
        embed = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])
        # Bi-Interaction Layer
        embed = 0.5 * (tf.pow(tf.reduce_sum(embed, axis=1), 2) -
                       tf.reduce_sum(tf.pow(embed, 2), axis=1))  # (None, embed_dim)
        # Concat
        x = tf.concat([dense_inputs, embed], axis=-1)
        # BatchNormalization
        x = self.bn(x, training=self.bn_use)
        # Hidden Layers
        x = self.dnn_network(x)
        outputs = tf.nn.sigmoid(self.dense(x))
        return outputs

    def summary(self):
        dense_inputs = Input(
            shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(
            shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs],
                    outputs=self.call([dense_inputs, sparse_inputs])).summary()
