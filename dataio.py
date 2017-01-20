"""IO functions.
"""


def build_vocab_index(fpath):
    with open(fpath) as fin:
        index = {line.strip(): i for i, line in enumerate(fin)}
    return index
