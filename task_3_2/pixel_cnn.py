import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfk = tf.keras
tfkl = tf.keras.layers
tfd = tfp.distributions


class PixelCNNConv(tfkl.Layer):
    def __init__(self, kernel_size, filters, type_A, use_bias=True, **kwargs):
        super(PixelCNNConv, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.filters = filters
        self.use_bias = use_bias
        self.type_A = type_A

        mask = np.zeros([self.kernel_size, self.kernel_size, 1, 1], dtype=np.float32)
        mask[:self.kernel_size // 2] = 1
        mask[self.kernel_size // 2, :self.kernel_size // 2] = 1
        if not self.type_A:
            mask[self.kernel_size // 2, self.kernel_size // 2] = 1

        self.mask = tf.constant(mask)

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel',
                                      shape=[self.kernel_size, self.kernel_size, input_shape[-1], self.filters],
                                      trainable=True,
                                      initializer=tfk.initializers.glorot_uniform)
        if self.use_bias:
            self.bias = self.add_weight('bias', shape=[1, 1, 1, self.filters], trainable=True,
                                        initializer=tfk.initializers.zeros)

    @tf.function
    def call(self, x):
        x = tf.nn.conv2d(x, filters=self.kernel * self.mask, strides=1, padding='SAME')
        if self.use_bias:
            x += self.bias
        return x


class ResidualBlock(tfkl.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.project = tfkl.Conv2D(kernel_size=1, filters=filters // 2, name='project')
        self.conv = PixelCNNConv(kernel_size=3, filters=filters // 2, type_A=False, name='pixel_cnn_conv')
        self.expand = tfkl.Conv2D(kernel_size=1, filters=filters, name='expand')

    @tf.function
    def call(self, x):
        shortcut = x

        x = self.project(x)
        x = self.conv(x)
        x = self.expand(x)

        return x + shortcut


class PixelCNNPrior(tfk.Model):
    def __init__(self, shape, **kwargs):
        super(tfk.Model, self).__init__(**kwargs)
        self.shape = shape

        self.params = tfk.Sequential([
            PixelCNNConv(filters=128, kernel_size=3, type_A=True, name='type_a_conv'),
            ResidualBlock(filters=128, name='block_1'),
            ResidualBlock(filters=128, name='block_2'),
            ResidualBlock(filters=128, name='block_3'),
            tfkl.Conv2D(filters=2, kernel_size=1, name='final_project')
        ])

        self.epsilon_prior = tfd.Independent(distribution=tfd.Normal(loc=tf.zeros(shape), scale=1.),
                                             reinterpreted_batch_ndims=3)

    @tf.function
    def calculate_logscale_and_translate(self, z):
        params = self.params(z)
        return tf.math.tanh(params[..., 0:1]), params[..., 1:2]

    @tf.function
    def log_prob(self, z):
        logscale, translate = self.calculate_logscale_and_translate(z)

        return self.epsilon_prior.log_prob((z - translate) / tf.math.exp(logscale)) - z.shape[-1] * tf.reduce_sum(
            logscale,
            axis=[1, 2, 3])

    @tf.function
    def sample(self, num_samples):
        eps = self.epsilon_prior.sample(num_samples)
        z = tf.zeros([num_samples] + self.shape)

        for _ in range(self.shape[0] * self.shape[1]):
            logscale, translate = self.calculate_logscale_and_translate(z)
            z = eps * tf.math.exp(logscale) + translate

        return z
