import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

class Embedding_model(tf.keras.Model):
    def __init__(self, emb_mat=None, mode="rand"):
        assert mode in ["rand", "static", "nonstatic"], \
                                "mode must be 'rand', 'static', 'nonstatic'"
        assert emb_mat is not None

        super(Embedding_model, self).__init__()
        self.mode = mode
        self.vocab_sz, self.embed_sz = emb_mat.shape

        if self.mode in ["rand", "static"]:
            embed_init = tf.initializers.constant(emb_mat)
            self.embed = tf.keras.layers.Embedding(
                                    self.vocab_sz,
                                    self.embed_sz,
                                    embeddings_initializer=embed_init
                          )
        elif self.mode == "nonstatic":
            self.embed = tf.keras.layers.Embedding(self.vocab_sz,
                                                    self.embed_sz)
            with tf.GradientTape() as tape:
                pred = self.embed(tf.convert_to_tensor([0]))
            tf.assign(
                        self.embed.variables[0],
                        tf.convert_to_tensor(emb_mat, dtype=tf.float32)
            )

    def call(self, input):
        return self.embed(input)

class CNN(tf.keras.Model):
    def __init__(
            self,
            filter_sz=[3, 4, 5],
            num_filters=[100, 100, 100],
            act=tf.keras.layers.ReLU(),
            dropout_rate=0.2,
            output_sz=5
    ):
        assert len(num_filters) == len(filter_sz)
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
