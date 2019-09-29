import os
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from coolname import generate_slug
from sklearn.model_selection import train_test_split
from real_nvp import AffineCouplingLayer, SigmoidFlow, LastChannelSplit
from glow import ActNorm

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tfk.layers


def sample_data(count=100000):
    rand = np.random.RandomState(0)
    a = [[-1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    b = [[1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    c = np.c_[2 * np.cos(np.linspace(0, np.pi, count // 3)),
              -np.sin(np.linspace(0, np.pi, count // 3))]
    c += rand.randn(*c.shape) * 0.2
    data_x = np.concatenate([a, b, c], axis=0)
    data_y = np.array([0] * len(a) + [1] * len(b) + [2] * len(c))
    perm = rand.permutation(len(data_x))

    return data_x[perm].astype(np.float32), data_y[perm]


def canvas_to_numpy(fig):
    fig.canvas.draw()
    cols, rows = fig.canvas.get_width_height()
    plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((rows, cols, 3))
    plt.close()

    return plot


class SmallRealNVP(tfk.Model):
    def _make_s_t(self, depth, width):
        layers = []
        for i in range(depth - 1):
            layers.append(tfkl.Dense(width, use_bias=False, activation=None))
            layers.append(tfkl.BatchNormalization())
            layers.append(tfkl.Activation('relu'))

        layers.append(tfkl.Dense(2, use_bias=True, activation=None))

        return tfk.Sequential(layers)

    def __init__(self, num_affine, parametrize_depth, parametrize_width, **kwargs):
        super(SmallRealNVP, self).__init__(**kwargs)

        self.num_affine = num_affine
        self.parametrize_depth = parametrize_depth
        self.parametrize_width = parametrize_width

        self.transform_layers = []

        for i in range(num_affine):
            self.transform_layers.append(ActNorm())
            self.transform_layers.append(
                AffineCouplingLayer(split=LastChannelSplit(flip=i % 2),
                                    s_t=self._make_s_t(parametrize_depth, parametrize_width)))

        self.transform_layers.append(ActNorm())
        self.transform_layers.append(SigmoidFlow())

        self.mesh_X, self.mesh_Y = tf.meshgrid(tf.linspace(-4., 4., 1000), tf.linspace(-4., 4., 1000))
        mesh = tf.stack([self.mesh_X, self.mesh_Y], axis=-1)
        self.mesh_flattened = tf.reshape(mesh, (-1, 2))

        grid_vertical_X, grid_vertical_Y = tf.meshgrid(tf.linspace(-4., 4., 25), tf.linspace(-4., 4., 1000))
        grid_horizontal_X, grid_horizontal_Y = tf.meshgrid(tf.linspace(-4., 4., 1000),
                                                           tf.linspace(-4., 4., 25))
        self.gridlines = tf.concat([tf.transpose(tf.stack([grid_vertical_X, grid_vertical_Y], axis=-1), [1, 0, 2]),
                                    tf.stack([grid_horizontal_X, grid_horizontal_Y], axis=-1)],
                                   axis=0)

    def _batched_function(self, data, function, batch_size):
        total_steps = np.ceil(len(data) / batch_size).astype(int)
        results = []
        for step in range(total_steps):
            indices = slice(step * batch_size, (step + 1) * batch_size)
            results.append(function(data[indices]))

        return tf.concat(results, axis=0)

    def transform(self, inputs):
        out = inputs
        for layer in self.transform_layers:
            out = layer(out)

        return out

    def log_prob(self, inputs):
        dets = []
        current = inputs
        for layer in self.transform_layers:
            dets.append(layer.log_determinant(current))
            current = layer(current)
            tf.summary.histogram(f'activations/{layer.name}', current)

        return tf.reduce_sum(tf.stack(dets, axis=1), axis=[1])

    def density_plot(self, eval_batch_size):
        log_probs = self._batched_function(self.mesh_flattened,
                                           function=self.log_prob,
                                           batch_size=eval_batch_size)

        fig = plt.figure()
        plt.pcolormesh(self.mesh_X, self.mesh_Y, np.exp(log_probs).reshape(self.mesh_X.shape))
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('Density $p(x_1, x_2)$')
        plt.colorbar()
        plt.tight_layout(pad=0)

        return canvas_to_numpy(fig)

    def latents_plot(self, eval_batch_size, X, Y):
        latents = self._batched_function(X,
                                         function=self.transform,
                                         batch_size=eval_batch_size)

        fig = plt.figure()
        plt.scatter(latents[:, 0], latents[:, 1], c=Y)
        plt.xlabel('$z_1$')
        plt.ylabel('$z_2$')
        plt.title('Mapped training data $f(x_1, x_2)$')
        plt.tight_layout(pad=0)
        return canvas_to_numpy(fig)

    def transformed_grid_plot(self, eval_batch_size):
        transformed_gridlines = self._batched_function(tf.reshape(self.gridlines, (-1, 2)),
                                                       function=self.transform,
                                                       batch_size=eval_batch_size)
        transformed_gridlines = tf.reshape(transformed_gridlines, self.gridlines.shape)

        fig = plt.figure()
        for gridline in transformed_gridlines:
            plt.plot(gridline[:, 0], gridline[:, 1], color='k')

        plt.xlabel('$z_1$')
        plt.ylabel('$z_2$')
        plt.title('$\mathcal{X}$ gridlines transformed into $\mathcal{Z}$')
        plt.tight_layout(pad=0)

        return canvas_to_numpy(fig)

    def fit(self, X, Y, val_X, epochs, optimizer, logdir, plot_freq, train_batch_size, eval_batch_size, val_freq):
        writer_train = tf.summary.create_file_writer(os.path.join(logdir, 'train'))
        writer_val = tf.summary.create_file_writer(os.path.join(logdir, 'val'))
        writer_images = tf.summary.create_file_writer(os.path.join(logdir, 'images'))
        writer_histograms = tf.summary.create_file_writer(os.path.join(logdir, 'histograms'))

        steps_per_epoch = np.ceil(len(X) / train_batch_size).astype(int)
        global_step = 0

        nll_loss = lambda X: tf.reduce_mean(-self.log_prob(X))

        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                tf.summary.experimental.set_step(global_step)
                random_indices = tf.random.uniform([train_batch_size], maxval=len(X), dtype=tf.int32)
                random_samples = tf.gather(X, random_indices)

                with tf.GradientTape() as tape:
                    with writer_histograms.as_default():
                        train_loss = nll_loss(random_samples)

                with writer_train.as_default():
                    tf.summary.scalar('loss', train_loss)

                grads = tape.gradient(train_loss, self.trainable_variables)
                with writer_histograms.as_default():
                    for grad, var in zip(grads, self.trainable_variables):
                        tf.summary.histogram(f'gradients/{var.name}', grad)

                optimizer.apply_gradients(zip(grads, self.trainable_variables))

                if global_step % val_freq == 0:
                    with writer_val.as_default():
                        val_loss = self._batched_function(val_X,
                                                          function=nll_loss,
                                                          batch_size=eval_batch_size)
                        tf.summary.scalar('loss', val_loss)

                if global_step % plot_freq == 0:
                    density = self.density_plot(eval_batch_size=eval_batch_size)
                    latents = self.latents_plot(eval_batch_size=eval_batch_size, X=X, Y=Y)
                    grid_plot = self.transformed_grid_plot(eval_batch_size=eval_batch_size)

                    together = np.concatenate([density, latents, grid_plot], axis=1)
                    with writer_images.as_default():
                        tf.summary.image('plots', together[np.newaxis, :, :, :])

                global_step += 1


def __main__():
    gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, enable=True)

    X, Y = sample_data()
    train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size=0.2)

    real_nvp = SmallRealNVP(num_affine=20,
                            parametrize_depth=10,
                            parametrize_width=128)

    adam = tfk.optimizers.Adam(1e-4)
    real_nvp.fit(X=train_X, Y=train_Y, val_X=val_X,
                 epochs=1000,
                 optimizer=adam,
                 logdir=f'logs/hw_2_1_2/{generate_slug()}',
                 plot_freq=10,
                 train_batch_size=4096,
                 val_freq=10,
                 eval_batch_size=32768)


if __name__ == '__main__':
    __main__()
