"""Train CNN classifier.
"""
import argparse
import json
import time
from pathlib import Path

import tensorflow as tf
import cnnmodel
import dataio

# training parameters
BATCH_SIZE = 50
NUM_EPOCHS = 3
KEEP_PROB = 0.8
CHECKPOINT_EVERY = 100

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
    utc_time = time.strftime('%Y%m%dT%H%M%S', time.gmtime())
    log_dir = Path(outdir, 'run-%s' % utc_time)
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    model_path = Path(log_dir, 'model.ckpt').as_posix()

    # loading data
    print('loading data...')
    start_time = time.time()
    vocab_index = dataio.build_vocab_index(args.vocabulary)
    train_data = dataio.load_data(args.train)
    dev_data = dataio.load_data(args.dev)
    print('data loaded [%.3f sec]' % (time.time() - start_time))

    # parameter inference
    vocab_size = len(vocab_index)
    sequence_length = len(train_data.sequences[0])
    num_classes = len(train_data.classes)
    # releasing resources
    del vocab_index
    print('vocabulary size:', vocab_size)

    # saving model parameters
    print('saving model parameters')
    parameters = {
        'sequence_length': sequence_length,
        'classes': train_data.classes,
        'vocab_size': vocab_size,
        'embedding_size': EMBEDDING_SIZE,
        'filter_sizes': FILTER_SIZES,
        'num_filters': NUM_FILTERS
    }
    parameter_file = log_dir / 'model-parameters.json'
    with parameter_file.open('w') as fout:
        json.dump(parameters, fout)

    # building the graph
    print('building graph')
    start_time = time.time()
    cnn = cnnmodel.TextCNN(sequence_length, num_classes, vocab_size,
                           EMBEDDING_SIZE, FILTER_SIZES, NUM_FILTERS)
    print('graph ready [%.3f sec]' % (time.time() - start_time))

    # savers
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter((log_dir / 'train').as_posix())
    test_writer = tf.summary.FileWriter((log_dir / 'test').as_posix())

    # test feed
    dev_feed = {
        cnn.keep_prob: 1.0,
        cnn.input_x: dev_data.sequences,
        cnn.input_y: dev_data.labels
    }

    # training
    print('starting training')
    with tf.Session() as sess:
        train_writer.add_graph(sess.graph)
        test_writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())

        step = 0
        for batch in dataio.batch_iter(
                train_data, BATCH_SIZE, NUM_EPOCHS, shuffle=True):
            step += 1
            sentences, labels = batch
            train_feed = {
                cnn.keep_prob: KEEP_PROB,
                cnn.input_x: sentences,
                cnn.input_y: labels
            }

            # running operations
            _, loss, accuracy, summaries = sess.run(
                [cnn.train_op, cnn.loss, cnn.accuracy, cnn.summaries],
                feed_dict=train_feed)
            train_writer.add_summary(summaries, step)

            if step % 10 == 0:
                print('Step %d, loss: %.4f, batch accuracy: %.4f' %
                      (step, loss, accuracy))

            if step % CHECKPOINT_EVERY == 0:
                saver.save(sess, model_path, global_step=step)
                accuracy, summaries = sess.run(
                    [cnn.accuracy, cnn.summaries],
                    feed_dict=dev_feed)
                test_writer.add_summary(summaries, step)
                print('Step %d, DEV accuracy: %.4f' % (step, accuracy))

        # saving last models
        saver.save(sess, model_path, global_step=step)


if __name__ == '__main__':
    main()
