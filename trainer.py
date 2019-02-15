import tensorflow as tf
import gzip
import numpy as np
from tqdm import tqdm
import os
from collections import Counter
import itertools
import random
import sys
import argparse
from pprint import pprint

import utils
from models import Net

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("id")
    parser.add_argument("-vec_file", action="store", dest="vec_file", \
                        default="data/GoogleNews-vectors-negative300.bin.gz",
                        help="Path to raw embeddings file")
    parser.add_argument("-emb_file", action="store", dest="emb_file", \
                        default="data/word2vec.npy",
                        help="Path to embeddings saved as np file")
    parser.add_argument("-epochs", action="store", dest="epochs", \
                        default=5, type=int)
    parser.add_argument("-dropout", action="store", dest="dropout_rate", \
                        default=0.5, type=float)
    parser.add_argument("-batch_size", action="store", dest="batch_size", \
                        default=32, type=int)
    parser.add_argument("-lr", action="store", dest="lr", \
                        default=1e-3, type=float)
    parser.add_argument("-lower", action="store", dest="lower_case", \
                        default=False, type=bool)
    parser.add_argument("-mode", action="store", dest="mode", \
                        default="rand", choices={"rand", "nonstatic"})
    parser.add_argument("-opt", action="store", dest="opt", \
                        default="Adam", choices={"Adam", "Adagrad", "Adadelta"})
    parser.add_argument("-hidden", action="store", dest="hidden", \
                        default=256, type=int)
    parser.add_argument("-model", action="store", dest="model", \
                        default=None)
    args = parser.parse_args()
    return args

def measure_confusion(sess, next_valid_elem, valid_preds, valid_iter_inits, num_labels=16):
    confusion = np.zeros((num_labels, num_labels))
    sess.run(valid_iter_inits)
    while True:
        try:
            label, pred = sess.run([next_valid_elem[1], valid_preds])
            confusion[label][pred[0]] += 1
        except tf.errors.OutOfRangeError:
            break
    return confusion

if __name__ == '__main__':
    args = parse_args()
    exp_id = args.id
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
    word2vec_filename = args.vec_file
    embed_filename = args.emb_file
    epochs = args.epochs
    dropout_rate = args.dropout_rate
    batch_size = args.batch_size
    learning_rate = args.lr
    mode = args.mode
    optimizer = args.opt
    load_file = args.model
    lower_case = args.lower_case

    padding = 0
    with open(logs_dir + "params", "w") as params_file:
        pprint(vars(args), params_file)

    train_X, train_y = utils.parse_file(data_dir + train_filename,
                                        lower_case=lower_case)
    valid_X, valid_y = utils.parse_file(data_dir + valid_filename,
                                        lower_case=lower_case)
    test_X = utils.parse_file(data_dir + test_filename, has_labels=False,
                              lower_case=lower_case)

    unk_index = utils.word2index["<UNK>"]

    if mode == "rand":
        embed_sz = 300
        embed_mat = np.random.uniform(-0.25, 0.25, size=(len(utils.word2index), embed_sz))
    else:
        if os.path.exists(embed_filename):
            known_word_embed = np.load(embed_filename)
            embed_sz = known_word_embed.shape[1] - 1
        else:
            known_word_embed, embed_sz = utils.save_bin_vec(
                                                utils.word2index,
                                                word2vec_filename,
                                                embed_filename
                                         )

        embed_mat = utils.make_embed_mat(len(utils.word2index), embed_sz,
                                            known_word_embed)
    embed_mat[0] = 0 

    binned_tr_X, binned_tr_y = utils.bin(train_X, train_y)

    batch_tr_dataset = []
    for i in range(len(binned_tr_X)):
        batch_tr_dataset.append(tf.data.Dataset.from_tensor_slices((
                        tf.convert_to_tensor(binned_tr_X[i], dtype=tf.int32),
                        tf.convert_to_tensor(binned_tr_y[i], dtype=tf.int32)
                        )).shuffle(
                                buffer_size=len(binned_tr_X[i])
                           ).batch(batch_size)
        )
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

    model = Net(embed_mat, mode=mode, dropout_rate=dropout_rate, hidden_sz=hidden_sz)

    train_logits = model.build_model(next_train_elem[0], True)
    train_loss_op = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=next_train_elem[1],
                        logits=train_logits
                    ))

    var_list=tf.trainable_variables()
    if optimizer == "Adam":
        optimizer_op = tf.train.AdamOptimizer(learning_rate).minimize(
                                            train_loss_op,
                                            var_list=var_list
                                        )
    elif optimizer == "Adagrad":
        optimizer_op = tf.train.AdagradOptimizer(learning_rate).minimize(
                                            train_loss_op,
                                            var_list=var_list
                                        )
    elif optimizer == "Adadelta":
        optimizer_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(
                                            train_loss_op,
                                            var_list=var_list
                                        )
    elif optimizer == "SGD":
        optimizer_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(
                                            train_loss_op,
                                            var_list=var_list
                                        )

    valid_logits = model.build_model(tf.expand_dims(next_valid_elem[0], 0), False)
    valid_loss_op = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.expand_dims(next_valid_elem[1], axis=0),
                        logits=valid_logits
                    ))
    valid_preds = tf.argmax(valid_logits, axis=1, output_type=tf.int32)
    valid_acc = tf.contrib.metrics.accuracy(tf.expand_dims(next_valid_elem[1], 0), valid_preds)
    orig_label = next_valid_elem[1]

    train_loss_file = open(logs_dir + "train_loss", "w")
    valid_logs = open(logs_dir + "valid_logs", "w")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #  measure_confusion(sess, next_valid_elem, valid_preds, valid_iter_inits)
    
    if load_file:
        model.load_model(sess, load_file)

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
        valid_logs.flush()
        if accuracy > max_acuracy:
            max_acuracy = accuracy
            model.save_model(sess, models_dir + "Epoch-%d-%f" % (n+1, accuracy))
