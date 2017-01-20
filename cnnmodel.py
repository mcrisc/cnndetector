"""Define the CNN model.
"""


import tensorflow as tf


class TextCNN:
    """Define the CNN model.

    This model implements the pattern inference/train/loss.

    Attributes
        predictions: computes the predictions
        loss: computes the loss
        train_op: executes the optimization
        accuracy: computes the accuracy in current batch
        summary: return summaries to be visualized in TensorBoard
    """

    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters):
        """Build the computation graph.
        """
        # input placeholders
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name='input_y')

        # embedding layer
        with tf.name_scope('embedding'), tf.device('/cpu:0'):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name='W')
            embeddings = tf.nn.embedding_lookup(W, self.input_x)
            # embeddings.shape = [batch, sequence_length, embedding_size]
            # adjusting dimensions
            conv_input = tf.expand_dims(embeddings, -1)
            # conv_input.shape = [batch, sequence_length, embedding_size, 1]

        # convolution layer
        with tf.name_scope('conv-layer'):
            conv_features = _conv_layer(filter_sizes, num_filters, conv_input)

        # dropout
        with tf.name_scope('dropout'):
            features_drop = tf.nn.dropout(conv_features, self.keep_prob)

        # output layer (fc-layer)
        with tf.name_scope('fc-layer'):
            self._scores = _fullyconnected_layer(features_drop, num_classes)
            self.predictions = tf.argmax(self._scores, 1)

        # loss
        with tf.name_scope('loss'):
            # cross entropy
            losses = tf.nn.softmax_cross_entropy_with_logits(
                self._scores, self.input_y)
            self.loss = tf.reduce_mean(losses)
            tf.summary.scalar('loss', self.loss)

        # training
        with tf.name_scope('optimize'):
            # optimizer
            optimizer = tf.train.AdamOptimizer()
            global_step = tf.Variable(0, name='global_step', trainable=False)
            self.train_op = optimizer.minimize(
                self.loss, global_step=global_step)

        # accuracy
        with tf.name_scope('accuracy'):
            # cross entropy
            correct_predictions = tf.equal(
                self.predictions,
                tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        # merging summaries
        self.summary = tf.summary.merge_all()


def _conv_filter(filter_size, num_filters, embedding_size, in_channels):
    """Creates a convolutional filter."""
    filter_shape = [filter_size, embedding_size, in_channels, num_filters]
    weights = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                          name='W')
    biases = tf.Variable(tf.constant(0.1, shape=[num_filters]),
                         dtype=tf.float32, name='b')
    return weights, biases


def _conv_layer(filter_sizes, num_filters, in_data):
    """Apply a set of convolutional filters over input data.

    :param in_data: a tensor of shape [batch, sequence_length,
    embedding_size, in_channels]
    :returns: a tensor of shape [batch, total_feature_maps]
    """
    sequence_length = in_data.get_shape()[1].value
    embedding_size = in_data.get_shape()[2].value
    in_channels = in_data.get_shape()[3].value

    # convolutional layer
    pooled_outputs = []
    for filter_size in filter_sizes:
        with tf.name_scope('conv-filter-%d' % filter_size):
            weights, biases = _conv_filter(
                filter_size, num_filters, embedding_size, in_channels)
            conv = tf.nn.conv2d(in_data, weights,
                                strides=[1, 1, 1, 1], padding='VALID')
            feature_map = tf.tanh(tf.nn.bias_add(conv, biases))
            # feature_map = tf.nn.relu(tf.nn.bias_add(conv, biases))
            height_conv = sequence_length - filter_size + 1
            pooled = tf.nn.max_pool(feature_map, ksize=[1, height_conv, 1, 1],
                                    strides=[1, 1, 1, 1], padding='VALID')
            # pooled.shape = [batch, 1, 1, num_filters]
            pooled_outputs.append(pooled)

    # concatenating pooled features
    features = tf.concat(3, pooled_outputs)
    total_feature_maps = len(filter_sizes) * num_filters
    features_flat = tf.reshape(features, [-1, total_feature_maps])
    # features_flat.shape = [batch, total_feature_maps]
    return features_flat


def _fullyconnected_layer(features, num_classes):
    """Build the fully-connected layer.

    :param features: tensor of shape [batch, num_features]
    :param num_classes: number of output classes
    """
    num_features = features.get_shape()[1].value

    weights = tf.Variable(tf.truncated_normal(
        [num_features, num_classes], stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[num_classes]),
                         dtype=tf.float32)
    scores = tf.nn.bias_add(tf.matmul(features, weights), biases)
    return scores


if __name__ == '__main__':
    # running simple tests
    import random
    vocab_size = 1000
    cnn = TextCNN(20, 3, vocab_size, 50, [1, 2, 3], 10)
    s1 = random.sample(range(vocab_size), 20)
    s2 = random.sample(range(vocab_size), 20)
    feed_dict = {cnn.keep_prob: 1.0,
                 cnn.input_x: [s1, s2],
                 cnn.input_y: [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]}
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        predictions, loss, accuracy = sess.run(
            [cnn.predictions, cnn.loss, cnn.accuracy],
            feed_dict=feed_dict)
    print('Test results')
    print('predictions:', predictions)
    print('loss:', loss)
    print('accuracy:', accuracy)
