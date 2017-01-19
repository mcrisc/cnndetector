"""Tokenize a dataset and exports the vocabulary.
"""
import argparse
import re
from pathlib import Path

RE_TOKENS = re.compile(r'\b\S+\b')
RE_DIGITS = re.compile(r'\d')


def load_data(fname):
    with open(fname) as fin:
        contents = [tuple(line.strip().split('\t')) for line in fin]
    return contents


def tokenize(data):
    """Tokenize sentences.
    """
    tokenized = [(RE_TOKENS.findall(RE_DIGITS.sub('0', sent)), label)
                 for sent, label in data]
    return tokenized


def compile_vocabulary(tokenized_data):
    vocab = set()
    for sent, _ in tokenized_data:
        vocab.update(sent)
    vocab = ['<PAD>'] + sorted(vocab)
    return vocab


def main():
    parser = argparse.ArgumentParser(description='Tokenize dataset')
    parser.add_argument(
        '-e', '--export-vocabulary', action='store_true',
        help='export vocabulary file')
    parser.add_argument('data', help='dataset file')
    parser.add_argument('outfile', help='basename of output file')
    args = parser.parse_args()

    print('tokenizing:', args.data)
    data = load_data(args.data)

    # writting tokenized file
    tokenized = tokenize(data)
    tokenized_file = Path(args.outfile).with_suffix('.tok')
    with tokenized_file.open('w') as fout:
        for tokens, label in tokenized:
            print(' '.join(tokens), label, sep='\t', file=fout)

    # writting vocabulary file
    if args.export_vocabulary:
        vocab = compile_vocabulary(tokenized)
        vocab_file = Path(args.outfile).with_suffix('.voc')
        with vocab_file.open('w') as fout:
            for word in vocab:
                print(word, file=fout)

    print('finished')


if __name__ == '__main__':
    main()
