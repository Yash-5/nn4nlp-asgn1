import tensorflow as tf
import gzip
import numpy as np
from tqdm import tqdm
import os
from collections import Counter
import itertools
import random
import sys

import utils
from models import Net

if __name__ == '__main__':
    exp_id = sys.argv[1]
    logs_dir = "./logs/" + exp_id + "/"
    models_dir = logs_dir + "models/"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    data_dir = "./data/"
    train_filename = "topicclass/topicclass_train.txt"
    valid_filename = "topicclass/topicclass_valid.txt"
    test_filename = "topicclass/topicclass_test.txt"
    word2vec_filename = "GoogleNews-vectors-negative300.bin.gz"

    embed_filename = "./data/word2vec.npy"
    epochs = 5
    padding = 0
    batch_size = 32
    learning_rate = 1e-3
    mode = "rand"
    optimizer = "Adam"
    load_file = None
    with open(logs_dir + "params", "a") as params_file:
        params_file.write("batch_size:" + str(batch_size) + "\n")
        params_file.write("learning_rate:" + str(learning_rate) + "\n")
        params_file.write("optimizer:" + optimizer + "\n")
        params_file.write("embed_filename:" + embed_filename + "\n")
        if load_file:
            params_file.write("load file:" + str(load_file) + "\n")
        params_file.write("mode:" + str(mode) + "\n\n")

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

    model = Net(embed_mat, mode=mode)

    train_logits = model.build_model(next_train_elem[0], True)
    train_loss_op = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=next_train_elem[1],
                        logits=train_logits
                    ))
    if optimizer == "Adam":
        optimizer_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(
                                            train_loss_op,
                                            var_list=tf.trainable_variables()
                                        )
    elif optimizer == "Adagrad":
        optimizer_op = tf.train.AdagradOptimizer(learning_rate).minimize(
                                            train_loss_op,
                                            var_list=tf.trainable_variables()
                                        )
    elif optimizer == "Adadelta":
        optimizer_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(
                                            train_loss_op,
                                            var_list=tf.trainable_variables()
                                        )

    valid_logits = model.build_model(tf.expand_dims(next_valid_elem[0], 0), False)
    valid_loss_op = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.expand_dims(next_valid_elem[1], axis=0),
                        logits=valid_logits
                    ))
    valid_preds = tf.argmax(valid_logits, axis=1, output_type=tf.int32)
    valid_acc = tf.contrib.metrics.accuracy(tf.expand_dims(next_valid_elem[1], 0), valid_preds)
    orig_label = next_valid_elem[1]

    train_loss_file = open(logs_dir + "train_loss", "a")
    valid_logs = open(logs_dir + "valid_logs", "a")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if load_file:
        self.model.load_model(sess, load_file)

    max_acuracy = 0

    for n in range(epochs):
        for i in tqdm(range(len(train_iter_inits)), leave=True):
            sess.run(train_iter_inits[i])
            while True:
                try:
                    train_loss, _ = sess.run([train_loss_op, optimizer_op])
                    train_loss_file.write(str(train_loss) + "\n")
                except tf.errors.OutOfRangeError:
                    break
        cor = 0
        valid_loss = []
        sess.run(valid_iter_inits)
        while True:
            try:
                t = sess.run([valid_acc, valid_loss_op])
                cor += t[0]
                valid_loss.append(t[1])
            except tf.errors.OutOfRangeError:
                break
        avg_valid_loss = sum(valid_loss) / len(valid_loss)
        accuracy = cor / len(valid_X)
        valid_logs.write("Epoch" + str(n + 1) + ": " + \
                            str(avg_valid_loss) + " " + str(accuracy) + "\n")
        if accuracy > max_acuracy:
            max_acuracy = accuracy
            model.save_model(sess, models_dir + "Epoch-%d-%f" % (n+1, accuracy))
