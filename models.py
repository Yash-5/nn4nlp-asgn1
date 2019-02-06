import tensorflow as tf

tf.enable_eager_execution()

class Embedding_model(tf.keras.Model):
    def __init__(self, W1, vocab1_sz=None, embed_sz=300, vocab2_sz=0,
                 W_init_val=0.25, mode="rand"):
        """
        For static and non static modes, infer vocab1_sz and embed_sz
        from given W1 and vocab2_sz needed to be non-negative

        For rand, vocab1_sz and embed_sz are needed and vocab2_sz should
        not be set
        """
        assert mode in ["rand", "static", "nonstatic"],
                                "mode must be 'rand', 'static', 'nonstatic'"
        if mode != "rand":
            assert vocab2_sz >= 0
            assert W2_init_val >= 0
            vocab1_sz, embed_sz = W1.shape
        else:
            assert vocab1_sz > 0, "vocab1_sz should be > 0"
            assert embed_sz > 0
            assert vocab2_sz == 0

        super(Model_em, self).__init__()
        self.mode = mode
        self.vocab1_sz = vocab1_sz
        self.embed_sz = embed_sz
        if self.mode == "rand":
            embed1_init = tf.initializers.random_uniform(minval=-W_init_val,
                                                         maxval=W_init_val)
            self.embed1 = tf.keras.layers.Embedding(
                                    self.vocab1_sz,
                                    self.embed_sz,
                                    embeddings_initializer=embed1_init
                          )
            return 
        elif self.mode == "static":
            pass

        self.embed1 = tf.keras.layers.Embedding(self.vocab1_sz, self.embed_sz)
        with tf.GradientTape() as tape:
            pred = self.embed1(tf.convert_to_tensor([0]))
        tf.assign(
                    self.embed1.variables[0],
                    tf.convert_to_tensor(W_em1, dtype=tf.float32)
        )

        embed2_init = tf.initializers.random_uniform(minval=-W2_init_val,
                                                     maxval=W2_init_val)
        self.embed2 = tf.keras.layers.Embedding(
                                self.vocab2_sz,
                                self.embed_sz,
                                embeddings_initializer=embed2_init
                      )
