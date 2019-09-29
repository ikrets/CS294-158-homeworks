import tensorflow as tf

tfk = tf.keras
tfkl = tf.keras.layers


class ActNorm(tfkl.Layer):
    def __init__(self, **kwargs):
        super(ActNorm, self).__init__(**kwargs)
        self.first_batch = True

    def _init(self, x):
        self.s = self.add_weight(shape=x.shape[-1],
                                 initializer='zeros',
                                 name=f'{self.name}/s',
                                 trainable=True)
        self.b = self.add_weight(shape=x.shape[-1],
                                 initializer='zeros',
                                 name=f'{self.name}/b',
                                 trainable=True)
        per_channel_mean = tf.reduce_mean(x, axis=tf.range(len(x.shape) - 1))
        per_channel_std = tf.math.reduce_std(x - per_channel_mean[tf.newaxis, :], axis=tf.range(len(x.shape) - 1))

        self.b.assign(-per_channel_mean / per_channel_std)
        self.s.assign(1. / per_channel_std)

        self.first_batch = False

    def call(self, x):
        if self.first_batch:
            self._init(x)

        return self.s * x + self.b

    def log_determinant(self, x):
        if self.first_batch:
            self._init(x)

        area = tf.reduce_prod(x.shape[1:-1])
        det = tf.cast(area, tf.float32) * tf.reduce_sum(tf.math.log(tf.abs(self.s)))
        return tf.broadcast_to(det, x.shape[0:1])
