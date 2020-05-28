import tensorflow as tf

rating = tf.feature_column.sequence_numeric_column('rating')
watches = tf.feature_column.sequence_categorical_column_with_identity('watches', num_buckets=1000)
watches_embedding = tf.feature_column.embedding_column(watches, dimension=10)
columns = [rating, watches_embedding]

sequence_input_layer = tf.keras.experimental.SequenceFeatures(columns)
features = tf.io.parse_example(..., features=tf.feature_column.make_parse_example_spec(columns))
sequence_input, sequence_length = sequence_input_layer(features)
sequence_length_mask = tf.sequence_mask(sequence_length)

rnn_cell = tf.keras.layers.SimpleRNNCell(33)
rnn_layer = tf.keras.layers.RNN(rnn_cell)
outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)
