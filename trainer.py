import tensorflow as tf
import gzip
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import utils

if __name__ == '__main__':
    data_dir = "./data/"
    train_filename = "topicclass/topicclass_train.txt"
    valid_filename = "topicclass/topicclass_valid.txt"
    test_filename = "topicclass/topicclass_test.txt"
    word2vec_filename = "GoogleNews-vectors-negative300.bin.gz"

    # Making sure UNK maps to 0
    embed_filename = "./data/word2vec.npy"

    train_X, train_y = utils.parse_file(data_dir + train_filename)
    valid_X, valid_y = utils.parse_file(data_dir + valid_filename)
    test_X = utils.parse_file(data_dir + test_filename, has_labels=False)

    if os.path.exists(embed_filename):
        known_word_embed = np.load(embed_filename)
        embed_sz = known_word_embed.shape[1] - 1
    else:
        known_word_embed, embed_sz = utils.save_bin_vec(
                                            utils.word2index,
                                            data_dir + word2vec_filename,
                                            embed_filename
                                     )

    embed_mat = utils.make_embed_mat(len(utils.word2index), embed_sz,
                                            known_word_embed)
    exit()
