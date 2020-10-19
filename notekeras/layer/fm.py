from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
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
