import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

class Embedding_model(tf.keras.Model):
    def __init__(self, emb1=None, vocab1_sz=None, embed_sz=300, vocab2_sz=0,
                 W_init_val=0.25, mode="rand"):
        """
        For static and non static modes, infer vocab1_sz and embed_sz
        from given emb1 and vocab2_sz needed to be non-negative

        For rand, vocab1_sz and embed_sz are needed and vocab2_sz should
        not be set
        """
        assert mode in ["rand", "static", "nonstatic"], \
                                "mode must be 'rand', 'static', 'nonstatic'"
        if mode != "rand":
            assert vocab2_sz >= 0
            assert W_init_val >= 0
            vocab1_sz, embed_sz = emb1.shape
        else:
            assert vocab1_sz > 0, "vocab1_sz should be > 0"
            assert embed_sz > 0
            assert emb1 is None
            assert vocab2_sz == 0

        super(Embedding_model, self).__init__()
        self.mode = mode
        if self.mode == "rand":
            self.vocab1_sz = vocab1_sz
            self.embed_sz = embed_sz
        else:
            self.vocab1_sz, self.embed_sz = emb1.shape
            self.vocab2_sz = vocab2_sz

        if self.mode == "rand":
            embed1_init = tf.initializers.random_uniform(minval=-W_init_val,
                                                         maxval=W_init_val)
            self.embed1 = tf.keras.layers.Embedding(
                                    self.vocab1_sz,
                                    self.embed_sz,
                                    embeddings_initializer=embed1_init
                          )
        elif self.mode == "static":
            embed1_init = tf.initializers.constant(emb1)
            self.embed1 = tf.keras.layers.Embedding(
                                    self.vocab1_sz,
                                    self.embed_sz,
                                    embeddings_initializer=embed1_init
                          )
            self.build_embed2(W_init_val)
        elif self.mode == "nonstatic":
            self.embed1 = tf.keras.layers.Embedding(self.vocab1_sz,
                                                    self.embed_sz)
            with tf.GradientTape() as tape:
                pred = self.embed1(tf.convert_to_tensor([0]))
            tf.assign(
                        self.embed1.variables[0],
                        tf.convert_to_tensor(emb1, dtype=tf.float32)
            )
            self.build_embed2(W_init_val)

    def build_embed2(self, W_init_val):
        embed2_init = tf.initializers.constant(np.random.uniform(
                                        low=-W_init_val,
                                        high=W_init_val,
                                        size=(self.vocab2_sz, self.embed_sz)
                                  ))
        self.embed2 = tf.keras.layers.Embedding(
                                self.vocab2_sz,
                                self.embed_sz,
                                embeddings_initializer=embed2_init
                      )
