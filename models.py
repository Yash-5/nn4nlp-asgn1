import tensorflow as tf
import numpy as np

class Embedding_model():
    def __init__(self, emb_mat):

        self.vocab_sz, self.embed_sz = emb_mat.shape

        self.emb_mat = tf.Variable(emb_mat, trainable=True)

    def build_model(self, input):
        h = tf.nn.embedding_lookup(self.emb_mat, input)
        h = h * tf.expand_dims(tf.cast(tf.sign(input), tf.float64), 2)
        return h

class CNN():
    def __init__(
            self,
            filter_sz,
            num_filters,
            act,
            dropout_rate,
            output_sz,
            name=""
    ):
        self.filter_sz = filter_sz
        self.num_filters = num_filters
        self.act = act
        self.dropout_rate = dropout_rate
        self.output_sz = output_sz
        self.name = name

    def build_model(self, input, training):
        h = []
        for i, (n, k) in enumerate(zip(self.num_filters, self.filter_sz)):
            h.append(tf.math.reduce_max(
                tf.layers.conv1d(
                    input,
                    filters=n,
                    kernel_size=k,
                    activation=self.act,
                    name=self.name + "conv" + str(i),
                    reuse=tf.AUTO_REUSE
                ),
                axis=1
            ))
        h = tf.concat(h, axis=1)
        h = tf.layers.dropout(
            h,
            rate=self.dropout_rate,
            training=training
        )
        h = tf.layers.dense(
                h,
                units=self.output_sz,
                activation=None,
                name=self.name + "fc",
                reuse=tf.AUTO_REUSE
            )
        return h

class Net():
    def __init__(
            self,
            emb_mat,
            mode="rand",
            filter_sz=[3, 4, 5],
            num_filters=[100, 100, 100],
            act=tf.nn.relu,
            dropout_rate=0.5,
            output_sz=16,
            max_to_keep=2
    ):
        assert len(num_filters) == len(filter_sz)
        self.emb_model = Embedding_model(emb_mat)
        self.max_to_keep = max_to_keep
        self.cnn_model = CNN(filter_sz, num_filters, act, dropout_rate, output_sz)

    def build_model(self, input, training):
        embed_out = self.emb_model.build_model(input)
        output = self.cnn_model.build_model(embed_out, training)
        self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES),
                                    max_to_keep=self.max_to_keep)
        return output

    def save_model(self, sess, save_file):
        self.saver.save(sess, save_file)

    def load_model(self, sess, save_file):
        self.saver.restore(sess, save_file)
