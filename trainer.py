import tensorflow as tf
import gzip
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from collections import Counter
import itertools
import random

import utils
from models import Net

if __name__ == '__main__':
    data_dir = "./data/"
    train_filename = "topicclass/topicclass_train.txt"
    valid_filename = "topicclass/topicclass_valid.txt"
    test_filename = "topicclass/topicclass_test.txt"
    word2vec_filename = "GoogleNews-vectors-negative300.bin.gz"

    embed_filename = "./data/word2vec.npy"
    epochs = 25
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

    binned_tr_X, binned_tr_y = utils.bin(train_X, train_y)

    for x, y in zip(binned_tr_X, binned_tr_y):
        print(len(x), len(y))

    batch_tr_dataset = []
    for i in range(len(binned_tr_X)):
        batch_tr_dataset.append(tf.data.Dataset.from_tensor_slices((
                        tf.convert_to_tensor(binned_tr_X[i], dtype=tf.int32),
                        tf.convert_to_tensor(binned_tr_y[i], dtype=tf.int32)
                        )).shuffle(
                                buffer_size=len(binned_tr_X[i])
                           ).batch(batch_size)
        )
        print(type(batch_tr_dataset[-1]))
    train_iter = tf.data.Iterator.from_structure((tf.int32, tf.int32),
                                                ([None, None], [None]))
    train_iter_inits = [train_iter.make_initializer(x) for x in \
                            batch_tr_dataset]
    next_train_elem = train_iter.get_next()

    valid_dataset = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(valid_X, valid_y),
                                                   output_types=(tf.int32, tf.int32),
                                                   output_shapes=([None], None))
    valid_iter = valid_dataset.make_one_shot_iterator()
    valid_iter_inits = valid_iter.make_initializer(valid_dataset)
    next_valid_elem = valid_iter.get_next()

    model = Net(embed_mat, mode="nonstatic")

    train_logits = model.build_model(next_train_elem[0], True)
    train_loss_op = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=next_train_elem[1],
                        logits=train_logits
                    ))
    optimizer_op = tf.train.AdadeltaOptimizer(0.003).minimize(train_loss_op, var_list=tf.trainable_variables())

    valid_logits = model.build_model(tf.expand_dims(next_valid_elem[0], 0), False)
    valid_loss_op = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=next_valid_elem[1],
                        logits=valid_logits
                   ))
    valid_preds = tf.argmax(valid_logits, axis=1, output_type=tf.int32)
    valid_acc = tf.contrib.metrics.accuracy(tf.expand_dims(next_valid_elem[1], 0), valid_preds)
    orig_label = next_valid_elem[1]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(valid_iter_inits)


    for n in range(epochs):
        for i in tqdm(range(len(train_iter_inits)), leave=True):
            sess.run(train_iter_inits[i])
            while True:
                try:
                    sess.run(optimizer_op)
                except tf.errors.OutOfRangeError:
                    break
        tot = 0
        cor = 0
        sess.run(valid_iter_inits)
        while True:
            try:
                t = sess.run([valid_acc])
                cor += t[0]
                tot += 1
            except tf.errors.OutOfRangeError:
                break
        print(cor, tot)
