import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                     Embedding, Flatten, Input, Layer)
from tensorflow.keras.regularizers import l2

from ...layers import FFM as FFM_Layer
from ...layers import Linear
from ...layers.fm import FactorizationMachine


class FM(keras.Model):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        super(FM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.fm = FactorizationMachine(output_dim=1)

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        sparse_input2 = tf.concat(
            [tf.one_hot(sparse_inputs[:, i],
                        depth=self.sparse_feature_columns[i]['feat_num'], dtype=tf.float32)
             for i in range(sparse_inputs.shape[1])
             ], axis=1)

        stack = tf.concat([dense_inputs, sparse_input2], axis=1)

        fm_outputs = self.fm(stack)
        outputs = tf.nn.sigmoid(fm_outputs)
        return outputs

    def summary(self, **kwargs):
        dense_inputs = keras.Input(name='iiiiii1',
                                   shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = keras.Input(name='iiiiii2',
                                    shape=(len(self.sparse_feature_columns),), dtype=tf.int32)

        keras.Model(inputs=[dense_inputs, sparse_inputs],
                    outputs=self.call([dense_inputs, sparse_inputs])).summary()


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
        dense_inputs, sparse_inputs = inputs
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
            x = tf.reduce_sum(element_wise_product, axis=1)   # (None, k)
        elif self.mode == 'avg':
            x = tf.reduce_mean(element_wise_product, axis=1)  # (None, k)
        else:
            x = self.attention(element_wise_product)  # (None, k)
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


class FFM(keras.Model):
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


class DeepFM(keras.Model):
    def __init__(self, feature_columns, k=10, hidden_units=(200, 200, 200), dnn_dropout=0.,
                 activation='relu', fm_w_reg=1e-4, fm_v_reg=1e-4, embed_reg=1e-4):
        """
        DeepFM
        :param feature_columns: A list. a list containing dense and sparse column feature information.
        :param k: A scalar. fm's latent vector number.
        :param hidden_units: A list. A list of dnn hidden units.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param activation: A string. Activation function of dnn.
        :param fm_w_reg: A scalar. The regularizer of w in fm.
        :param fm_v_reg: A scalar. The regularizer of v in fm.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(DeepFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }

        self.fm = FactorizationMachine(
            output_dim=1, factor_dim=k, kernal_reg=fm_v_reg, weight_reg=fm_w_reg,                                           name='FMM')

        self.dnn = DNN(hidden_units, activation, dnn_dropout)
        self.dense = Dense(1, activation=None)
        self.w1 = self.add_weight(name='wide_weight',
                                  shape=(1,),
                                  trainable=True)
        self.w2 = self.add_weight(name='deep_weight',
                                  shape=(1,),
                                  trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(1,),
                                    trainable=True)

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=-1)
        stack = tf.concat([dense_inputs, sparse_embed], axis=-1)
        # wide
        wide_outputs = self.fm(stack)
        # deep
        deep_outputs = self.dnn(stack)
        deep_outputs = self.dense(deep_outputs)

        outputs = tf.nn.sigmoid(
            tf.add(tf.add(self.w1 * wide_outputs, self.w2 * deep_outputs), self.bias))
        return outputs

    def summary(self):
        dense_inputs = Input(name='i111',
                             shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(name='i222',
                              shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        keras.Model(name='deep-fm', inputs=[dense_inputs, sparse_inputs], outputs=self.call(
            [dense_inputs, sparse_inputs])).summary()


class CIN(layers.Layer):
    """
    CIN part
    """

    def __init__(self, cin_size, l2_reg=1e-4):
        """

        :param cin_size: A list. [H_1, H_2 ,..., H_k], a list of the number of layers
        :param l2_reg: A scalar. L2 regularization.
        """
        super(CIN, self).__init__()
        self.cin_size = cin_size
        self.l2_reg = l2_reg

    def build(self, input_shape):
        # get the number of embedding fields
        self.embedding_nums = input_shape[1]
        # a list of the number of CIN
        self.field_nums = [self.embedding_nums] + self.cin_size
        # filters
        self.cin_W = {
            'CIN_W_' + str(i): self.add_weight(
                name='CIN_W_' + str(i),
                shape=(
                    1, self.field_nums[0] * self.field_nums[i], self.field_nums[i + 1]),
                initializer='random_uniform',
                regularizer=l2(self.l2_reg),
                trainable=True)
            for i in range(len(self.field_nums) - 1)
        }

    def call(self, inputs, **kwargs):
        dim = inputs.shape[-1]
        hidden_layers_results = [inputs]
        # split dimension 2 for convenient calculation
        # dim * (None, field_nums[0], 1)
        split_X_0 = tf.split(hidden_layers_results[0], dim, 2)
        for idx, size in enumerate(self.cin_size):
            # dim * (None, filed_nums[i], 1)
            split_X_K = tf.split(hidden_layers_results[-1], dim, 2)

            # (dim, None, field_nums[0], field_nums[i])
            result_1 = tf.matmul(split_X_0, split_X_K, transpose_b=True)

            result_2 = tf.reshape(
                result_1, shape=[dim, -1, self.embedding_nums * self.field_nums[idx]])

            # (None, dim, field_nums[0] * field_nums[i])
            result_3 = tf.transpose(result_2, perm=[1, 0, 2])

            result_4 = tf.nn.conv1d(input=result_3, filters=self.cin_W['CIN_W_' + str(idx)], stride=1,
                                    padding='VALID')

            # (None, field_num[i+1], dim)
            result_5 = tf.transpose(result_4, perm=[0, 2, 1])

            hidden_layers_results.append(result_5)

        final_results = hidden_layers_results[1:]
        # (None, H_1 + ... + H_K, dim)
        result = tf.concat(final_results, axis=1)
        result = tf.reduce_sum(result,  axis=-1)  # (None, dim)

        return result


class xDeepFM(keras.Model):
    def __init__(self, feature_columns, hidden_units, cin_size, dnn_dropout=0, dnn_activation='relu',
                 embed_reg=1e-5, cin_reg=1e-5):
        """
        xDeepFM
        :param feature_columns: A list. a list containing dense and sparse column feature information.
        :param hidden_units: A list. a list of dnn hidden units.
        :param cin_size: A list. a list of the number of CIN layers.
        :param dnn_dropout: A scalar. dropout of dnn.
        :param dnn_activation: A string. activation function of dnn.
        :param embed_reg: A scalar. the regularizer of embedding.
        :param cin_reg: A scalar. the regularizer of cin.
        """
        super(xDeepFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_dim = self.sparse_feature_columns[0]['embed_dim']
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.linear = Linear()
        self.cin = CIN(cin_size=cin_size, l2_reg=cin_reg)
        self.dnn = DNN(hidden_units=hidden_units,
                       dnn_dropout=dnn_dropout, dnn_activation=dnn_activation)
        self.cin_dense = Dense(1)
        self.dnn_dense = Dense(1)
        self.bias = self.add_weight(name='bias', shape=(
            1, ), initializer=tf.zeros_initializer())

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs
        # linear  delete
        # linear_out = self.linear(sparse_inputs)

        embed = [self.embed_layers['embed_{}'.format(i)](
            sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        # cin
        embed_matrix = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])
        cin_out = self.cin(embed_matrix)  # (None, embedding_nums, dim)
        cin_out = self.cin_dense(cin_out)
        # dnn
        embed_vector = tf.reshape(
            embed_matrix, shape=(-1, embed_matrix.shape[1] * embed_matrix.shape[2]))
        dnn_out = self.dnn(embed_vector)
        dnn_out = self.dnn_dense(dnn_out)

        output = tf.nn.sigmoid(cin_out + dnn_out + self.bias)
        return output

    def summary(self):
        dense_inputs = Input(
            shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(
            shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call(
            [dense_inputs, sparse_inputs])).summary()
