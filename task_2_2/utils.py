import tensorflow as tf

@tf.function
def preprocess(x, n_bits):
    x = tf.cast(x, tf.float32)
    x += tf.random.uniform(x.shape, minval=0, maxval=1.)
    x = .05 + .95 * x / (2 ** n_bits)

    return x


@tf.function
def postprocess(x, n_bits):
    x = (x - .05) * (2 ** n_bits) / .95
    x = tf.cast(x, tf.uint8) * tf.cast(256 / (2 ** n_bits), tf.uint8)

    return x
