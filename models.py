import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

class Embedding_model(tf.keras.Model):
    def __init__(self, emb_mat, mode="rand"):

        super(Embedding_model, self).__init__()
        self.mode = mode
        self.vocab_sz, self.embed_sz = emb_mat.shape

        if self.mode in ["static"]:
            self.emb_mat = tf.contrib.eager.Variable(emb_mat, trainable=False)
        elif self.mode in ["rand", "nonstatic"]:
            self.emb_mat = tf.contrib.eager.Variable(emb_mat, trainable=True)

    def call(self, input):
        return tf.nn.embedding_lookup(self.emb_mat, input)

class CNN(tf.keras.Model):
    def __init__(
            self,
            filter_sz,
            num_filters,
            act,
            dropout_rate,
            output_sz
    ):
        super(CNN, self).__init__()
        
        self.dropout_rate = dropout_rate
        self.conv_layers = []

        for c, k in zip(num_filters, filter_sz):
            self.conv_layers.append(tf.keras.layers.Conv1D(
                filters=c,
                kernel_size=k,
                activation=act,
                padding="same"
            ))
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.dense = tf.keras.layers.Dense(output_sz)

    def call(self, input, training=False):
        h = []
        for c in self.conv_layers:
            h.append(tf.math.reduce_max(c(input), axis=1))
        h = tf.concat(h, axis=1)
        if training:
            h = self.dropout(h, training=training)
        h = self.dense(h)
        return h

class Net(tf.keras.Model):
    def __init__(
            self,
            emb_mat,
            mode="rand",
            filter_sz=[3, 4, 5],
            num_filters=[100, 100, 100],
            act=tf.keras.layers.ReLU(),
            dropout_rate=0.2,
            output_sz=16
    ):
        assert mode in ["rand", "static", "nonstatic"], \
                                "mode must be 'rand', 'static', 'nonstatic'"
        assert len(num_filters) == len(filter_sz)
        super(Net, self).__init__()
        self.emb_model = Embedding_model(emb_mat, mode)
        self.cnn_model = CNN(filter_sz, num_filters, act, dropout_rate, output_sz)

    def call(self, input, training=False):
        h = self.emb_model(input)
        h = self.cnn_model(h, training=True)
        return h
