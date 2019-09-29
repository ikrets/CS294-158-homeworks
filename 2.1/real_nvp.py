import tensorflow as tf
import numpy as np

tfkl = tf.keras.layers


class AffineCouplingLayer(tfkl.Layer):
    def __init__(self, split, s_t, **kwargs):
        super(AffineCouplingLayer, self).__init__(**kwargs)

        self.split = split
        self.s_t = s_t

    def log_determinant(self, x):
        x1, _ = self.split.split(x)
        s_t = self.s_t(x1)
        s = s_t[..., :s_t.shape[-1] // 2]
        return tf.reduce_sum(tf.math.log(tf.math.sigmoid(s)),
                             axis=tf.range(1, len(x.shape)))

    def call(self, x):
        x1, x2 = self.split.split(x)
        s_t = self.s_t(x1)
        s = s_t[..., :s_t.shape[-1] // 2]
        t = s_t[..., s_t.shape[-1] // 2:]

        return self.split.combine(x1, (x2 * tf.math.sigmoid(s) + t))


class SigmoidFlow(tfkl.Layer):
    def __init__(self, **kwargs):
        super(SigmoidFlow, self).__init__(**kwargs)

    def call(self, x):
        return tf.math.sigmoid(x)

    def log_determinant(self, x):
        dsigmoid = tf.math.sigmoid(x) * (1 - tf.math.sigmoid(x))
        dsigmoid = tf.clip_by_value(dsigmoid, 1e-10, 1.)
        return tf.reduce_sum(tf.math.log(dsigmoid), axis=tf.range(1, len(x.shape)))


class LastChannelSplit:
    def __init__(self, flip):
        self.flip = flip

    def split(self, x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return (x1, x2) if not self.flip else (x2, x1)

    def combine(self, x1, x2):
        return tf.concat([x1, x2] if not self.flip else [x2, x1], axis=-1)
