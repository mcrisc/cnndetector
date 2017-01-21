"""Detect language variants.
"""
import argparse
import json
import time
from pathlib import Path

import tensorflow as tf
import cnnmodel
import dataio


def main():
    parser = argparse.ArgumentParser(description='Detect language variants.')
    parser.add_argument('modeldir', help='directory of trained model')
    parser.add_argument('data', help='data to classify')
    args = parser.parse_args()

    # preparing output directories
    model_dir = Path(args.modeldir)

    # loading data
    print('loading data...')
    start_time = time.time()
    data = dataio.load_data(args.data)
    elapsed = time.time() - start_time
    print('data loaded [%.3f sec]' % elapsed)

    # loading parameters
    print('loading parameters')
    parameter_file = model_dir / 'model-parameters.json'
    with parameter_file.open() as fin:
        params = json.load(fin)

    # building the graph
    print('building graph')
    start_time = time.time()
    cnn = cnnmodel.TextCNN(params['sequence_length'], len(params['classes']),
                           params['vocab_size'], params['embedding_size'],
                           params['filter_sizes'], params['num_filters'])
    elapsed = time.time() - start_time
    print('graph ready [%.3f sec]' % elapsed)

    # savers
    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(model_dir.as_posix())

    # data feed
    data_feed = {
        cnn.keep_prob: 1.0,
        cnn.input_x: data.sequences,
        cnn.input_y: data.labels
    }

    # classification
    with tf.Session() as sess:
        print('restoring checkpoint:', latest_checkpoint)
        saver.restore(sess, latest_checkpoint)

        print('starting classification')
        start_time = time.time()
        predictions, accuracy = sess.run(
            [cnn.predictions, cnn.accuracy],
            feed_dict=data_feed)
        elapsed = time.time() - start_time
        print('classification finished [%.3f sec]' % elapsed)

    print('predictions:', predictions)
    print('accuracy: %.4f' % accuracy)


if __name__ == '__main__':
    main()
