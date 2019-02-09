import tensorflow as tf
import gzip
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from collections import Counter

import random

import utils
from models import Net

tf.enable_eager_execution()

if __name__ == '__main__':
    data_dir = "./data/"
    train_filename = "topicclass/topicclass_train.txt"
    valid_filename = "topicclass/topicclass_valid.txt"
    test_filename = "topicclass/topicclass_test.txt"
    word2vec_filename = "GoogleNews-vectors-negative300.bin.gz"

    embed_filename = "./data/word2vec.npy"
    epochs = 10
    padding = 0
    batch_size = 32

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

    batch_tr_X, batch_tr_y = utils.bin(train_X, train_y)
    for x, y in zip(batch_tr_X, batch_tr_y):
        print(len(x), len(y))

    batch_tr = []
    for i in range(len(batch_tr_X)):
        batch_tr.append(tf.data.Dataset.from_tensor_slices((
                            tf.convert_to_tensor(batch_tr_X[i], dtype=tf.int32),
                            tf.convert_to_tensor(batch_tr_y[i], dtype=tf.int32)
                        )).shuffle(
                                buffer_size=len(batch_tr_X[i])
                           ).batch(batch_size)
        )
        print(type(batch_tr[-1]))

    model = Net(embed_mat, mode="nonstatic")
    optimizer = tf.train.AdadeltaOptimizer(0.001)

    for i in range(len(valid_X)):
        valid_X[i] = tf.convert_to_tensor(valid_X[i:i+1], dtype=tf.int32)
        valid_y[i] = tf.convert_to_tensor(valid_y[i:i+1], dtype=tf.int32)
    
    for n in tqdm(range(epochs), leave=False):
        for i in tqdm(range(len(batch_tr)), leave=False):
            for (x, y) in batch_tr[i]:
                with tf.GradientTape() as tape:
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=y,
                                logits=model(x)
                           )
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
        mean_loss = 0
        accuracy = 0
        for i in range(len(valid_X)):
            logits=model(valid_X[i])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=valid_y[i],
                        logits=logits
                   )
            accuracy += int(tf.argmax(logits, axis=1).numpy()[0] == \
                                    valid_y[i].numpy()[0])
            mean_loss += loss.numpy()
        mean_loss /= len(valid_X)
        accuracy /= len(valid_X)
        print("Loss:", mean_loss)
        print("Accuracy:", accuracy)
