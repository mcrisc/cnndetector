"""Map words to integer indices.
"""
import argparse

import dataio

SENTENCE_SIZE = 80


def main():
    parser = argparse.ArgumentParser(
        description='Map words to integer indices')
    parser.add_argument('vocabulary', help='vocabulary file')
    parser.add_argument('tokenized', help='tokenized dataset file')
    parser.add_argument('outfile', help='basename of output file')
    args = parser.parse_args()

    # building vocabulary
    vocab_index = dataio.build_vocab_index(args.vocabulary)
    pad = vocab_index['<PAD>']

    # mapping words to indices
    with open(args.tokenized) as fin, open(args.outfile, 'w') as fout:
        for line in fin:
            sentence, label = line.strip().split('\t')
            tokens = sentence.split()[:SENTENCE_SIZE]  # truncating
            indices = [vocab_index[token] for token in tokens
                       if token in vocab_index]
            indices.extend([pad] * (SENTENCE_SIZE - len(indices)))
            print(' '.join(map(str, indices)), label, sep='\t', file=fout)
    print('finished')


if __name__ == '__main__':
    main()
