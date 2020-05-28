import tensorflow as tf
from tensorflow.keras.layers import DenseFeatures, Input, Embedding, Layer
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import sequence_feature_column as sfc

from notekeras.feature import IndicatorColumnDef

field_type_map = {
    'int': tf.dtypes.int32,
    'int32': tf.dtypes.int32,
    'int64': tf.dtypes.int64,
    'float': tf.dtypes.float32,
    'float32': tf.dtypes.float32,
    'float64': tf.dtypes.float64,
    'string': tf.dtypes.string,
}


###################################################################################################################


def _sequence_cate_share_embedding_column(params):
    feature = sfc.sequence_categorical_column_with_vocabulary_list(params['key'], params['vocabulary'])

    shared_embedding_column = fc.shared_embedding_columns_v2([feature],
                                                             dimension=params['dimension'],
                                                             shared_embedding_collection_name=params['share_name'])

    return shared_embedding_column


class ParseFeatureConfig:
    def __init__(self):
        self.feature_dict = {}
        self.share_layer = {}

    def _get_input_layer(self, params: dict, size=1) -> (str, Layer):
        field_name = params['key']
        if 'length' in params.keys():
            size = params['length']
        field_type = field_type_map.get(params['dtype'], tf.dtypes.string)

        if field_name in self.feature_dict.keys():
            inputs = self.feature_dict[field_name]
        else:
            inputs = Input((size,), dtype=field_type, name=field_name)
            self.feature_dict[field_name] = inputs

        return field_name, inputs

    def _get_share_layer(self, name: dict, layer: Layer) -> Layer:
        """
        根据名称取出共享层
        :param name:
        :param layer:
        :return:
        """
        if name is None:
            return layer
        elif name in self.share_layer.keys():
            return self.share_layer[name]
        else:
            self.share_layer[layer.name] = layer
            return layer

    def _numeric_column(self, params: dict) -> Layer:
        key, inputs = self._get_input_layer(params)

        return inputs

    def _bucketized_column(self, params: dict) -> DenseFeatures:
        key, inputs = self._get_input_layer(params)

        feature = fc.numeric_column(params['key'])
        feature_column = fc.bucketized_column(feature, boundaries=params['boundaries'])

        outputs = DenseFeatures(feature_column, name=params.get('name', None))({key: inputs})
        return outputs

    @staticmethod
    def _get_categorical_column(params: dict) -> fc.CategoricalColumn:
        if 'vocabulary' in params.keys():
            feature = fc.categorical_column_with_vocabulary_list(params['key'], params['vocabulary'], default_value=0)
        elif 'bucket_size' in params.keys():
            feature = fc.categorical_column_with_hash_bucket(params['key'], hash_bucket_size=params['bucket_size'])
        elif 'file' in params.keys():
            feature = fc.categorical_column_with_vocabulary_file(params['key'], vocabulary_file=params['file'])
        elif 'num_buckets' in params.keys():
            feature = fc.categorical_column_with_identity(params['key'], num_buckets=params['num_buckets'])
        elif 'boundaries' in params.keys():
            feature = fc.bucketized_column(fc.numeric_column(params['key']), boundaries=params['boundaries'])
        else:
            raise Exception("params error")

        return feature

    @staticmethod
    def _get_sequence_categorical_column(params: dict) -> fc.SequenceCategoricalColumn:
        key = params['key']
        if 'vocabulary' in params.keys():
            feature = sfc.sequence_categorical_column_with_vocabulary_list(key, params['vocabulary'], default_value=0)
        elif 'bucket_size' in params.keys():
            feature = sfc.sequence_categorical_column_with_hash_bucket(key, hash_bucket_size=params['bucket_size'])
        elif 'file' in params.keys():
            feature = sfc.sequence_categorical_column_with_vocabulary_file(key, vocabulary_file=params['file'])
        elif 'num_buckets' in params.keys():
            feature = sfc.sequence_categorical_column_with_identity(key, num_buckets=params['num_buckets'])
        else:
            raise Exception("params error")

        return feature

    def _cate_embedding_column(self, params: dict) -> Layer:
        key, inputs = self._get_input_layer(params)

        feature = self._get_categorical_column(params)

        column = IndicatorColumnDef(feature, size=1)

        sequence_input = DenseFeatures(column)({key: inputs})
        sequence_input = tf.keras.backend.sum(sequence_input, axis=-1)

        name = params.get('share_name', None)
        layer = self._get_share_layer(name,
                                      Embedding(input_dim=feature.num_buckets,
                                                output_dim=params['dimension'],
                                                mask_zero=True,
                                                name=name))
        res = layer(sequence_input)
        return res

    def _cate_indicator_column(self, params: dict) -> DenseFeatures:
        key, inputs = self._get_input_layer(params)

        feature = self._get_categorical_column(params)
        feature_column = fc.indicator_column(feature)

        outputs = DenseFeatures(feature_column, name=params.get('name', None))({key: inputs})

        return outputs

    def _sequence_cate_embedding_column(self, params: dict):
        key, inputs = self._get_input_layer(params, size=params['length'])

        feature = self._get_sequence_categorical_column(params)
        column = IndicatorColumnDef(feature, size=params['length'])

        sequence_input, sequence_length = sfc.SequenceFeatures(column)({key: inputs})

        sequence_input = tf.keras.backend.sum(sequence_input, axis=-1)

        name = params.get('share_name', None)
        layer = self._get_share_layer(name,
                                      Embedding(input_dim=len(params['vocabulary']),
                                                output_dim=params['dimension'],
                                                mask_zero=True,
                                                name=name))
        res = layer(sequence_input)
        return res, sequence_length

    def _get_columns_map(self, key: str):
        _columns_map = {
            "NumericColumn": self._numeric_column,
            "BucketizedColumn": self._cate_indicator_column,

            "CateIndicatorColumn": self._cate_indicator_column,
            "FileIndicatorColumn": self._cate_indicator_column,
            "HashIndicatorColumn": self._cate_indicator_column,
            "BucketIndicatorColumn": self._cate_indicator_column,

            "CateEmbeddingColumn": self._cate_embedding_column,
            "FileEmbeddingColumn": self._cate_embedding_column,
            "HashEmbeddingColumn": self._cate_embedding_column,
            "BucketEmbeddingColumn": self._cate_embedding_column,

            "SequenceCateEmbddingColumn": self._sequence_cate_embedding_column,
            "SequenceFileEmbddingColumn": self._sequence_cate_embedding_column,
            "SequenceHashEmbddingColumn": self._sequence_cate_embedding_column
        }
        return _columns_map.get(key, None)

    def parse_feature_json(self, layer_json) -> Layer:
        outputs = []
        for feature_line in layer_json["inputs"]:
            feature_type_name = feature_line['type']
            feature_para = feature_line['parameters']

            method = self._get_columns_map(feature_type_name)
            if method is None:
                continue

            outputs.append(method(feature_para))

        outputs = tf.keras.backend.concatenate(outputs)

        return outputs

    def parse_sequence_feature_json(self, layer_json):
        feature_line = layer_json['inputs'][0]

        feature_type_name = feature_line['type']
        feature_para = feature_line['parameters']

        method = self._get_columns_map(feature_type_name)
        if method is None or not isinstance(feature_para, dict):
            raise Exception("error")

        sequence_input, sequence_length = method(feature_para)
        return sequence_input, sequence_length
