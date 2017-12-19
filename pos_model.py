import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import data_utils
import sys


def main():
    PRETRAIN_EMBEDDINGS = False
    USE_SUBWORDS = True
    if len(sys.argv) > 1:
        PRETRAIN_EMBEDDINGS = sys.argv[1]
    if len(sys.argv) > 2:
        USE_SUBWORDS = sys.argv[2]

    print('pre-trained embeddings is set to %s' % PRETRAIN_EMBEDDINGS)
    print('sub-words embeddings is set to %s' % USE_SUBWORDS)

    EPOCHS_TO_TRAIN = 100
    CONTEXT_SIZE = 5
    BATCH_SIZE = 1000
    WORKERS = 2

    vocab = data_utils.read_vocab(data_utils.vocab_path)
    embeddings = np.random.randn(len(vocab), 50) / np.sqrt(len(vocab))
    EMBEDDING_DIM = len(embeddings[0])
    vocab_reversed = {}
    if PRETRAIN_EMBEDDINGS:
        embeddings = data_utils.read_embeddings(data_utils.wv_path)
    if USE_SUBWORDS:
        embeddings, vocab, vocab_reversed = data_utils.generate_embeddings_with_prefixes(embeddings, vocab, EMBEDDING_DIM)

    print('Starting execution...')
    print('using EMBEDDING_DIM of %s' % EMBEDDING_DIM)
    print('using CONTEXT_SIZE of %s' % CONTEXT_SIZE)
    print('using BATCH_SIZE of %s' % BATCH_SIZE)
    print('using EPOCHS_TO_TRAIN of %s' % EPOCHS_TO_TRAIN)

    class Net(nn.Module):

        def __init__(self, vocab_size, embed_dim, context_size, pretrained_embeddings):
            super(Net, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embed_dim)
            self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            # if PRETRAIN_EMBEDDINGS:
            #     self.embeddings.weight.requires_grad = False
            self.linear1 = nn.Linear(context_size * embed_dim, 128)
            self.linear2 = nn.Linear(128, len(data_utils.POS_TAGS))

        def forward(self, x):
            embeds = self.embeddings(x).view((-1,CONTEXT_SIZE * EMBEDDING_DIM))

            if USE_SUBWORDS:
                prefixes = get_prefixes_embeddings(x, vocab, vocab_reversed)
                suffixes = get_suffixes_embeddings(x, vocab, vocab_reversed)

                prefixes_tensor = Variable((torch.from_numpy(prefixes)).type(torch.LongTensor))
                suffixes_tensor = Variable((torch.from_numpy(suffixes)).type(torch.LongTensor))

                prefixes_embeds = self.embeddings(prefixes_tensor).view((-1,CONTEXT_SIZE * EMBEDDING_DIM))
                suffixes_embeds = self.embeddings(suffixes_tensor).view((-1, CONTEXT_SIZE * EMBEDDING_DIM))

                embeds = embeds + prefixes_embeds + suffixes_embeds

            out = F.tanh(self.linear1(embeds))
            out = F.log_softmax(self.linear2(out))
            return out

    net = Net(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, embeddings)

    print('Preparing train/test/dev sets')
    train_data_loader = data_utils.prepare_tensor_dataset(data_utils.pos_train, vocab, WORKERS, BATCH_SIZE)
    dev_data_loader = data_utils.prepare_tensor_dataset(data_utils.pos_dev, vocab, WORKERS, BATCH_SIZE)
    test_data_loader = data_utils.prepare_tensor_dataset(data_utils.pos_test, vocab, WORKERS, BATCH_SIZE, include_y=False)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01)
    dev_losses = []
    train_losses = []
    acceptances = []
    iterations = []
    print("Starting training loop")
    for idx in range(0, EPOCHS_TO_TRAIN):
        for iteration, batch in enumerate(train_data_loader, 1):
            x, y = Variable(batch[0]), Variable(batch[1])
            optimizer.zero_grad()
            output = net(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        if idx % 1 == 0:
            # calculate accuracy on validation set
            dev_loss = 0
            net.eval()
            correct = 0.0
            total = 0.0
            for dev_batch_idx, dev_batch in enumerate(dev_data_loader):
                x, y = Variable(dev_batch[0]), Variable(dev_batch[1])
                output = net(x)
                dev_loss = criterion(output, y)
                _, predicted = torch.max(output.data, 1)
                total += dev_batch[1].size(0)
                correct += (predicted == dev_batch[1]).sum()

            acc = correct / total

            acceptances.append(acc)
            train_losses.append(loss.data[0])
            dev_losses.append(dev_loss.data[0])
            iterations.append(idx)
            print("Epoch {: >8}     TRAIN_LOSS: {: >8}      DEV_LOSS: {: >8}     ACC: {}".format(idx, loss.data[0], dev_loss.data[0], acc))

    print("Predicting the test file")
    net.eval()

    test_file = open(os.path.join(data_utils.POS_DIR, "test_results.txt"), 'w')

    for test_batch_idx, test_batch in enumerate(test_data_loader):
        x, y = Variable(test_batch[0]), Variable(test_batch[1])
        output = net(x)
        predictions = torch.max(output.data, 1)[1].numpy()
        for pos_index in predictions:
            test_file.write(data_utils.get_pos_name_by_index(pos_index) + "\n")

    test_file.close()
    print('Finished execution!')

    print('Plotting graphs..')
    fig = plt.figure()
    fig.suptitle("POS - random word vectors with prefix/suffix", fontsize=14)
    ax = plt.subplot(211)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Train loss')
    ax.plot(iterations, train_losses, 'k')

    ax = plt.subplot(212)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Dev loss')
    ax.plot(iterations, dev_losses, 'k')

    ax = plt.subplot(213)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Acc')
    ax.plot(iterations, acceptances, 'k')
    plt.show()


def get_prefixes_embeddings(x, vocab, vocab_reversed):
    arr = np.empty(shape=x.size(), dtype=int)
    for i in range(x.size()[0]):
        prefixes = [get_prefix_for_word_index(w_index, vocab, vocab_reversed) for w_index in x[i]]
        arr[i] = np.asarray(prefixes, dtype=int)
    return arr


def get_prefix_for_word_index(w_index, vocab, vocab_reversed):
    index = w_index.data[0]
    w = vocab_reversed[index]
    w_len = len(w)
    if w_len > 3:
        prefix = w[:3]
        if prefix in vocab_reversed:
            return vocab[prefix]
        else:
            return vocab["uuunkkk"]
    else:
        return index


def get_suffixes_embeddings(x, vocab, vocab_reversed):
    arr = np.empty(shape=x.size(), dtype=int)
    for i in range(x.size()[0]):
        suffixes = [get_suffix_for_word_index(w_index, vocab, vocab_reversed) for w_index in x[i]]
        arr[i] = np.asarray(suffixes, dtype=int)
    return arr


def get_suffix_for_word_index(w_index, vocab, vocab_reversed):
    index = w_index.data[0]
    w = vocab_reversed[index]
    w_len = len(w)
    if w_len > 3:
        suffix = w[w_len - 3:]
        if suffix in vocab_reversed:
            return vocab[suffix]
        else:
            return vocab["uuunkkk"]
    else:
        return index



if __name__ == "__main__":
    main()