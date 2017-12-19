import os
from itertools import chain
import torch

import numpy as np
from torch.utils.data import TensorDataset, DataLoader

NER_TAGS = {'PER' : 0, 'LOC' : 1, 'OGR' : 2, 'TIME': 3, 'O': 4}
POS_TAGS = {'CC' : 0, 'CD' : 1, 'DT' : 2, 'EX' : 3, 'FW' : 4, 'IN' : 5, 'JJ' : 6, 'JJR' : 7, 'JJS' : 8, 'LS' : 9, 'MD' : 10, 'NN' : 11, 'NNS' : 12, 'NNP' : 13, 'NNPS' : 14, 'PDT' : 15, 'POS' : 16, 'PRP' : 17, 'PRP$' : 18, 'RB' : 19, 'RBR' : 20, 'RBS' : 21, 'RP' : 22, 'SYM' : 23, 'TO' : 24, 'UH' : 25, 'VB' : 26, 'VBD' : 27, 'VBG' : 28, 'VBN' : 29, 'VBP' : 30, 'VBZ' : 31, 'WDT' : 32, 'WP' : 33, 'WP$' : 34, 'WRB' : 35}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
POS_DIR = os.path.join(ROOT_DIR, 'pos')
WORDS_DIR = os.path.join(ROOT_DIR, 'words')

pos_train = os.path.join(POS_DIR, "train")
pos_dev = os.path.join(POS_DIR, "dev")
pos_test = os.path.join(POS_DIR, "test")


vocab_path = os.path.join(WORDS_DIR, "vocab.txt")
wv_path = os.path.join(WORDS_DIR, "wordVectors.txt")

def get_pos_name_by_index(index):
    for k, v in POS_TAGS.iteritems():
        if v == index:
            return k
    print('error: no pos value found for index %s' %index)
    return ''

def get_pos_tag_vector(pos_value):
    # y = np.zeros(len(POS_TAGS), dtype=int)
    # if pos_value in POS_TAGS:
    #     y[POS_TAGS[pos_value]] = 1
    # return y
    if pos_value in POS_TAGS:
        return POS_TAGS[pos_value]
    else:
        return 0


def prepare_data(fname, vocab, include_y=True):
    sentences = read_sentences(fname, include_y)
    data = []
    for s in sentences:
        sentence_data = [([s[i][0], s[i + 1][0], s[i + 2][0], s[i + 3][0], s[i + 4][0]], s[i + 2][1])
            for i in range(len(s) - 4)]
        data.append(sentence_data)
    data = np.asarray(data)
    data = list(chain.from_iterable(data))
    X = [d[0] for d in data]

    X_indexes = []

    for window in X:
        window_indexes = []
        for word in window:
            if (word in vocab):
                window_indexes.append(vocab[word])
            else:
                window_indexes.append(vocab['uuunkkk'])
        X_indexes.append(window_indexes)

    if include_y:
        Y = [d[1] for d in data]
        Y_indexes = [get_pos_tag_vector(pos) for pos in Y]

    else:
        Y_indexes = [0]*len(data)
    return np.asarray(X_indexes), np.asarray(Y_indexes)


def read_sentences(fname, include_y=True):
    sentences = []
    with open(fname) as f:
        content = f.readlines()

    sentence = []
    sentence.append(('start', "NONE"))
    sentence.append(('start', "NONE"))
    for line in content:
        if line != '\n':
            line = line.strip()
            word = line.split()[0].lower()
            pos = ""
            if include_y:
                pos = line.split()[1]
            sentence.append((word, pos))
        else:
            sentence.append(('end', "NONE"))
            sentence.append(('end', "NONE"))
            sentences.append(sentence)
            sentence = []

    return sentences


def read_vocab(fname):
    with open(fname) as f:
        content = f.readlines()
    vocab = {k.strip().lower(): v for v, k in enumerate(content)}
    return vocab


def read_embeddings(fname):
    with open(fname) as f:
        content = f.readlines()

    vecs = [np.fromstring(w_vec, dtype=float, sep=' ') for w_vec in content]
    return np.asarray(vecs)


def prepare_tensor_dataset(fname, vocab, workers, batch_size, include_y=True):
    X, Y = prepare_data(fname, vocab, include_y)

    X = (torch.from_numpy(X)).type(torch.LongTensor)
    Y = (torch.from_numpy(Y)).type(torch.LongTensor)

    train_set = TensorDataset(X, Y)
    training_data_loader = DataLoader(dataset=train_set, num_workers=workers, batch_size=batch_size, shuffle=True)
    return training_data_loader


def generate_embeddings_with_prefixes(embeddings, vocab, embed_dim):
    vocab_index = len(vocab)
    print('generating embeddings with prefixes..')
    print('original vocab length: %s' % vocab_index)
    print('original embeddings size: %s %s' % (embeddings.shape[0], embeddings.shape[1]))

    new_words = []
    new_vocab = vocab.copy()
    vocab_reversed = {}
    for w in vocab:
        w_len = len(w)
        vocab_reversed[vocab[w]] = w
        if w_len > 3:
            prefix = w[:3]
            suffix = w[w_len - 3:]
            if prefix not in new_vocab:
                new_words.append(prefix)
                new_vocab[prefix] = vocab_index
                vocab_reversed[vocab_index] = prefix
                vocab_index += 1

            if suffix not in new_vocab:
                new_words.append(suffix)
                new_vocab[suffix] = vocab_index
                vocab_reversed[vocab_index] = suffix
                vocab_index += 1

    new_embeddings = np.random.randn(len(new_words), embed_dim) / np.sqrt(len(new_words))
    new_embeddings = np.append(embeddings, new_embeddings, axis=0)

    print('updated vocab length: %s' % len(new_vocab))
    print('updated embeddings size: %s %s' % (new_embeddings.shape[0], new_embeddings.shape[1]))

    return new_embeddings, new_vocab, vocab_reversed



