from itertools import chain
import nltk


import numpy as np

import data_utils

vocab = data_utils.read_vocab(data_utils.vocab_path)

X, Y = data_utils.prepare_data(data_utils.pos_train, vocab)

# pos = nltk.pos_tag()
# print pos

# X_indexes = []
# for window in X:
#     window_indexes = []
#     for word in window:
#         if(word in vocab):
#             window_indexes.append(vocab[word])
#         else:
#             window_indexes.append(vocab['UUUNKKK'])
#     X_indexes.append(window_indexes)
#
print Y[:10]
# print X_indexes[0]
# embeddings = data_utils.read_embeddings(data_utils.wv_path)



# print(embeddings[:2])


# train = data_utils.read_sentences(data_utils.pos_train)
# print train[0]
#
# trigrams = [([train[0][i][0], train[0][i + 1][0], train[0][i + 2][0], train[0][i + 3][0], train[0][i + 4][0]], train[0][i + 2][1])
#             for i in range(len(train[0]) - 4)]

# print (trigrams)