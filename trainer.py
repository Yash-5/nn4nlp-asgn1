import tensorflow as tf
import gzip
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

delimiter = "|||"

data_dir = "./data/"
train_filename = "topicclass/topicclass_train.txt"
valid_filename = "topicclass/topicclass_valid.txt"
test_filename = "topicclass/topicclass_test.txt"
word2vec_filename = "GoogleNews-vectors-negative300.bin.gz"

label2index = defaultdict(lambda: len(label2index))
word2index = defaultdict(lambda: len(label2index))

def load_bin_vec(vocab, fname=""):
    known_word_vecs = {}
    with gzip.open(fname, 'rb') as w2vfile:
        header = w2vfile.readline()
        vocab_sz, embed_sz = map(int, header.split())
        embed_readlen = np.dtype('float32').itemsize * embed_sz
        for v in tqdm(range(vocab_sz), leave=False):
            word = b''
            while True:
                c = w2vfile.read(1)
                if c == b' ':
                    break
                if c != b'\n':
                    word += c
            word = word.decode('utf-8')
            embedding = w2vfile.read(embed_readlen)
            if word in vocab:
                known_word_vecs[vocab[word]] = np.frombuffer(embedding,
                                                             dtype='float32')
    return known_word_vecs, embed_sz

def calc_mean_and_std(known_word_vecs):
    temp = np.array([vec for i, vec in known_word_vecs.items()])
    return np.mean(temp, axis=0), np.std(temp, axis=0)

def parse_file(fname, has_labels=True):
    global label2index
    global word2index
    X = []
    if has_labels:
        y = []
    with open(fname, 'r') as f:
        for line in tqdm(f, leave=False):
            X.append([])
            label, words = line.split(delimiter)
            if has_labels:
                y.append(label2index[label.strip()])
            words = words.strip().split()
            for word in words:
                X[-1].append(word2index[word])
    if has_labels:
        return X, y
    return X

train_X, train_y = parse_file(data_dir + train_filename)
print(len(train_X), len(train_y))
valid_X, valid_y = parse_file(data_dir + valid_filename)
print(len(valid_X), len(valid_y))
test_X = parse_file(data_dir + test_filename)
print(len(word2index))

def bin_stats():
    total = [0 for i in range(7)]
    m, M = 100, 0
    for x in train_X:
        total[(len(x) + 9)//10] += 1
        m, M = min(m, len(x)), max(len(x), M)
    print("min is %d, max is %d" % (m, M))
    print(total)

bin_stats()
exit()

plt.hist([len(x) for x in train_X])
plt.savefig("histgram of train lengths.png")
exit()

known_word_vecs, embed_sz = load_bin_vec(word2index,
                                         data_dir + word2vec_filename)
known_vec_mean, known_vec_std = calc_mean_and_std(known_word_vecs)
print(known_vec_mean.shape, known_vec_std.shape)

vec_at_i = lambda i : known_word_vecs[i] if i in known_word_vecs \
                        else np.random.randn(embed_sz) * known_vec_std + \
                                    known_vec_mean

word2vec_mat = np.zeros(shape=(len(word2index), embed_sz))
for i in range(len(word2index)):
    word2vec_mat[i] = vec_at_i(i)
print(word2vec_mat.shape)
print(np.where(word2vec_mat == 0))
