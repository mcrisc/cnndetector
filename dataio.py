"""IO functions.
"""
from collections import namedtuple
import numpy as np

DataSet = namedtuple('DataSet', ['sequences', 'labels', 'classes'])


def build_vocab_index(fpath):
    with open(fpath) as fin:
        vocab_index = {line.strip(): i for i, line in enumerate(fin)}
    return vocab_index


def _shuffle_batch(sequences, labels):
    data_size = len(sequences)
    ix = np.random.permutation(data_size)
    return (sequences[ix, :], labels[ix, :])


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Generates a batch iterator for a dataset.
    """
    sequences, labels = data.sequences, data.labels
    if shuffle:
        sequences, labels = _shuffle_batch(sequences, labels)
    for _ in range(num_epochs):
        begin = 0
        while begin < len(data.sequences):
            end = begin + batch_size
            batch = (sequences[begin: end], labels[begin: end])
            if shuffle:
                batch = _shuffle_batch(*batch)
            yield batch
            begin = end


def load_data(fpath):
    sequences = []
    labels = []
    classes = []
    with open(fpath) as fin:
        for line in fin:
            sentence, label = line.strip().split('\t')
            if label not in classes:
                classes.append(label)
            sequence = np.array(
                list(map(int, sentence.split())), dtype=np.int32)
            sequences.append(sequence)
            labels.append(label)

    sequence_arr = np.array(sequences)
    labels_arr = np.zeros((len(sequences), len(classes)), dtype=np.float32)
    # one-hot encoding
    for i, label in enumerate(labels):
        labels_arr[i, classes.index(label)] = 1.0

    return DataSet(sequence_arr, labels_arr, tuple(classes))
