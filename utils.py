import gzip
import numpy as np
from tqdm import tqdm
from collections import defaultdict

label2index = defaultdict(lambda: len(label2index))
word2index = defaultdict(lambda: len(word2index))
word2index["<UNK>"]

def save_bin_vec(vocab, fname, save_name):
    known_word_vecs = []
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
                known_word_vecs.append(np.append([vocab[word]],
                                    np.frombuffer(embedding, dtype='float32')))
    known_word_vecs = np.array(known_word_vecs)
    known_word_vecs = np.sort(known_word_vecs, axis=0)
    np.save(save_name, known_word_vecs)
    return known_word_vecs, embed_sz

def bin_stats(train_X):
    total = [0 for i in range(20)]
    m, M = 100, 0
    for x in train_X:
        total[(len(x) + 9)//10] += 1
        m, M = min(m, len(x)), max(len(x), M)
    print("min is %d, max is %d" % (m, M))
    print(total)

def parse_file(fname, has_labels=True, delimiter="|||"):
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

def make_embed_mat(vocab_sz, embed_sz, known_word_embed):
    global word2index
    known_vec_mean, known_vec_std = np.mean(known_word_embed[:, 1:], axis=0), \
                                    np.std(known_word_embed[:, 1:], axis=0)

    rand_emb = lambda : np.random.randn(embed_sz) * known_vec_std + \
                            known_vec_mean
    embed_mat = np.zeros(shape=(len(word2index), embed_sz))
    j = 0
    tot = 0
    for i in range(1, embed_mat.shape[0]):
        if j < known_word_embed.shape[0] and i == int(known_word_embed[j, 0]):
            embed_mat[i] = known_word_embed[j, 1:]
            j += 1
        else:
            embed_mat[i] = rand_emb()
    return embed_mat
