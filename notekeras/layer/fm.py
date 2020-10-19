from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2


class FactorizationMachine(Layer):
    """
    """

    def __init__(self,
                 output_dim=30,
                 factor_dim=10,
                 name='FM',
                 activation="relu",
                 use_weight=True,
                 use_bias=True,
                 *args,
                 **kwargs):
        """
        """
        super(FactorizationMachine, self).__init__(name=name, *args, **kwargs)
        self.output_dim = output_dim
        self.factor_dim = factor_dim
        self.activate = activation
        self.user_weight = use_weight
        self.use_bias = use_bias
        self.weight = self.bias = self.kernel = None
        self.activate_layer = None

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.factor_dim),
                                      initializer='glorot_uniform',
                                      regularizer=l2(1e-4),
                                      trainable=True)
        if self.user_weight:
            self.weight = self.add_weight(name='weight',
                                          shape=(
                                              input_shape[1], self.output_dim),
                                          initializer='glorot_uniform',
                                          regularizer=l2(1e-4),
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


class FMLayer(Layer):
    def __init__(self, output_dim,
                 factor_order,
                 activation=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FMLayer, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.factor_order = factor_order
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.w = self.add_weight(name='one',
                                 shape=(input_dim, self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.v = self.add_weight(name='two',
                                 shape=(input_dim, self.factor_order),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='bias',
                                 shape=(self.output_dim,),
                                 initializer='zeros',
                                 trainable=True)

        super(FMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        X_square = K.square(inputs)

        xv = K.square(K.dot(inputs, self.v))
        xw = K.dot(inputs, self.w)

        p = 0.5 * K.sum(xv - K.dot(X_square, K.square(self.v)), 1)
        rp = K.repeat_elements(K.reshape(p, (-1, 1)), self.output_dim, axis=-1)

        f = xw + rp + self.b

        output = K.reshape(f, (-1, self.output_dim))

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.output_dim
