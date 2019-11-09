import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tf.keras.layers

tfd = tfp.distributions
tfpl = tfp.layers


def residual_stack(x):
    for i in range(5):
        shortcut = x

        x = tfkl.Activation('relu')(x)
        x = tfkl.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        x = tfkl.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = tfkl.Conv2D(filters=256, kernel_size=3, padding='same', activation=None)(x)
        x = tfkl.Lambda(lambda X: tf.math.sigmoid(X[0]) * X[1])([shortcut, x])

    return tfkl.Activation('relu')(x)


def encoder(x):
    x = tfkl.Conv2D(filters=128, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = tfkl.Conv2D(filters=256, kernel_size=4, strides=2, padding='same', activation='relu')(x)
    x = tfkl.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation=None)(x)
    x = residual_stack(x)
    x = tfkl.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation=None)(x)

    def distribution(X):
        d = tfd.Normal(loc=X[..., :X.shape[-1] // 2],
                       scale=tf.math.exp(tf.math.tanh(X[..., X.shape[-1] // 2:])))
        return tfd.Independent(distribution=d,
                               reinterpreted_batch_ndims=3)

    return tfpl.DistributionLambda(make_distribution_fn=distribution)(x)


def decoder(x, fixed_std=None):
    x = tfkl.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation=None)(x)
    x = residual_stack(x)

    x = tfkl.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same', activation='relu')(x)

    x = tfkl.Conv2DTranspose(filters=3 + (3 if fixed_std is None else 0), kernel_size=4, strides=2, padding='same',
                             activation=None)(x)

    def distribution(X):
        if fixed_std is not None:
            loc = X
            scale = fixed_std
        else:
            loc = X[..., X.shape[-1] // 2:]
            scale = tf.math.exp(tf.math.tanh(X[..., X.shape[-1] // 2:]))

        return tfd.Independent(distribution=tfd.Normal(loc=loc, scale=scale),
                               reinterpreted_batch_ndims=3)

    return tfpl.DistributionLambda(make_distribution_fn=distribution)(x)
