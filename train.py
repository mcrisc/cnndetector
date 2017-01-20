"""Train CNN classifier.
"""
import argparse
import time
from pathlib import Path

import tensorflow as tf
import cnnmodel
import dataio

# training parameters
BATCH_SIZE = 50
NUM_EPOCHS = 3
KEEP_PROB = 0.8

# model parameters
EMBEDDING_SIZE = 100
FILTER_SIZES = [1, 2, 3]
NUM_FILTERS = 100


def main():
    parser = argparse.ArgumentParser(description='Train the CNN classifier')
    parser.add_argument('vocabulary', help='vocabulary file')
    parser.add_argument('train', help='training data')
    parser.add_argument('dev', help='validation data')
    parser.add_argument(
        'outdir', help='output directory to save models and logs')
    args = parser.parse_args()

    # preparing output directories
    outdir = Path(args.outdir)
    if not outdir.exists():
        outdir.mkdir(parents=True)
    model_dir = Path(outdir, 'models')
    log_dir = Path(outdir, 'logs')
    for directory in (model_dir, log_dir):
        if not directory.exists():
            directory.mkdir()

    # loading data
    print('loading data...')
    start_time = time.time()
    vocab_index = dataio.build_vocab_index(args.vocabulary)
    train_data = dataio.load_data(args.train)
    # dev_data = dataio.load_data(args.dev)
    print('data loaded [%.3f sec]' % (time.time() - start_time))

    # parameter inference
    vocab_size = len(vocab_index)
    sequence_length = len(train_data.sequences[0])
    num_classes = len(train_data.classes)
    # releasing resources
    del vocab_index
    print('vocabulary size:', vocab_size)

    # building the graph
    print('building graph')
    start_time = time.time()
    cnn = cnnmodel.TextCNN(sequence_length, num_classes, vocab_size,
                           EMBEDDING_SIZE, FILTER_SIZES, NUM_FILTERS)
    print('graph ready [%.3f sec]' % (time.time() - start_time))

    # savers
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(log_dir.as_posix())

    # training
    print('starting training')
    with tf.Session() as sess:
        train_writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())

        step = 0
        for batch in dataio.batch_iter(
                train_data, BATCH_SIZE, NUM_EPOCHS, shuffle=True):
            step += 1
            sentences, labels = batch
            feed_dict = {
                cnn.keep_prob: KEEP_PROB,
                cnn.input_x: sentences,
                cnn.input_y: labels
            }

            # running operations
            _, loss, accuracy, summaries = sess.run(
                [cnn.train_op, cnn.loss, cnn.accuracy, cnn.summaries],
                feed_dict=feed_dict)
            train_writer.add_summary(summaries, step)

            if step % 10 == 0:
                print('Step %d, loss: %.4f, batch accuracy: %.4f' %
                      (step, loss, accuracy))

            if step > 50:
                break


if __name__ == '__main__':
    main()
