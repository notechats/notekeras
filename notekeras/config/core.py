import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras.layers import DenseFeatures, Input

field_type_map = {
    'int': tf.dtypes.int32,
    'int32': tf.dtypes.int32,
    'int64': tf.dtypes.int64,
    'float': tf.dtypes.float32,
    'float32': tf.dtypes.float32,
    'float64': tf.dtypes.float64,
    'string': tf.dtypes.string,
}


def parse_feature_json(feature_json):
    feature_dict = {}
    feature_columns = []
    for feature_line in feature_json:
        feature_type_name = feature_line['name']
        feature_para = feature_line['parameters']

        field_name = feature_para['input_tensor']
        field_type = field_type_map.get(feature_para['dtype'], tf.dtypes.string)
        feature_dict[field_name] = Input((1,), dtype=field_type, name=field_name)

        if feature_type_name == 'NumericColumn':
            feature_columns.append(feature_column.numeric_column(feature_para['input_tensor']))
        elif feature_type_name == 'BucketizedColumn':
            feature = feature_column.numeric_column(feature_para['input_tensor'])
            feature_columns.append(feature_column.bucketized_column(feature, boundaries=feature_para['boundaries']))
        elif feature_type_name == 'IndicatorColumn':
            feature = feature_column.categorical_column_with_vocabulary_list(feature_para['input_tensor'],
                                                                             feature_para['vocabulary'])
            feature_columns.append(feature_column.indicator_column(feature))
        elif feature_type_name == 'EmbeddingColumn':
            feature = feature_column.categorical_column_with_vocabulary_list(feature_para['input_tensor'],
                                                                             feature_para['vocabulary'])
            feature_columns.append(feature_column.embedding_column(feature, feature_para['dimension']))
        elif feature_type_name == 'CrossedFeature':
            feature = feature_column.categorical_column_with_vocabulary_list(feature_para['input_tensor'],
                                                                             feature_para['vocabulary'])
            feature_columns.append(feature_column.embedding_column(feature, feature_para['dimension']))

        else:
            print(feature_type_name)

    feature_layer = DenseFeatures(feature_columns)

    return feature_layer(feature_dict), feature_dict
