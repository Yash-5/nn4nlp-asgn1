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
    parser.add_argument("-dropout", action="store", dest="dropout_rate", \
                        default=0.5, type=float)
    parser.add_argument("-lower", action="store", dest="lower_case", \
                        default=False, type=bool)
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
    logs_dir = "./predict/" + exp_id + "/"
    models_dir = logs_dir + "models/"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    data_dir = "./data/"
    train_filename = "topicclass/topicclass_train.txt"
    valid_filename = "topicclass/topicclass_valid.txt"
    test_filename = "topicclass/topicclass_test.txt"
    dropout_rate = args.dropout_rate
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
    index2label = {}
    for w, i in utils.label2index.items():
        index2label[i] = w

    embed_sz = 300
    embed_mat = np.random.uniform(-0.25, 0.25, size=(len(utils.word2index), embed_sz))

    embed_mat[0] = 0 

    valid_dataset = tf.data.Dataset.from_generator(lambda: valid_X,
                                                   output_types=tf.int32,
                                                   output_shapes=[None])
    valid_iter = valid_dataset.make_one_shot_iterator()
    valid_iter_inits = valid_iter.make_initializer(valid_dataset)
    next_valid_elem = valid_iter.get_next()

    test_dataset = tf.data.Dataset.from_generator(lambda: test_X,
                                                  output_types=tf.int32,
                                                  output_shapes=[None])
    test_iter = test_dataset.make_one_shot_iterator()
    test_iter_inits = test_iter.make_initializer(test_dataset)
    next_test_elem = test_iter.get_next()

    model = Net(embed_mat, dropout_rate=dropout_rate)

    test_logits = model.build_model(tf.expand_dims(next_test_elem, 0), False)
    test_preds = tf.argmax(test_logits, axis=1, output_type=tf.int32)

    valid_logits = model.build_model(tf.expand_dims(next_valid_elem, 0), False)
    valid_preds = tf.argmax(valid_logits, axis=1, output_type=tf.int32)

    valid_pred_logs = open(logs_dir + "valid_preds", "w")
    test_pred_logs = open(logs_dir + "test_preds", "w")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if load_file:
        model.load_model(sess, load_file)

    while True:
        try:
            t = sess.run(valid_preds)
            valid_pred_logs.write(index2label[t[0]] + "\n")
        except tf.errors.OutOfRangeError:
            break
    while True:
        try:
            t = sess.run(test_preds)
            test_pred_logs.write(index2label[t[0]] + "\n")
        except tf.errors.OutOfRangeError:
            break
