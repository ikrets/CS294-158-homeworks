import os
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from coolname import generate_slug
from sklearn.model_selection import train_test_split

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

    return data_x[perm], data_y[perm]


def canvas_to_numpy(fig):
    fig.canvas.draw()
    cols, rows = fig.canvas.get_width_height()
    plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((rows, cols, 3))
    plt.close()

    return plot


class ParametrizeGaussianMixtureBlock(tfkl.Layer):
    def __init__(self, num_components, depth, width, conditioned=False, **kwargs):
        super(ParametrizeGaussianMixtureBlock, self).__init__(**kwargs)

        self.depth = depth
        self.width = width
        self.num_components = num_components
        self.conditioned = conditioned

        self.first_input = self.add_weight(shape=(1, width), initializer='glorot_normal', trainable=True)
        self.concatenate = tfkl.Concatenate(name='concatenate_condition')

        self.layers = [tfkl.Dense(width, activation='relu', name=f'dense_{i}') for i in range(self.depth)]
        self.layers.append(tfkl.Dense(num_components * 3, name='params'))
        self.layers = tfk.Sequential(self.layers)

        self.stds = tfkl.Lambda(tf.math.square, name='stds')

    @tf.function
    def call(self, inputs):
        if self.conditioned:
            if inputs.shape.rank == 1:
                inputs = inputs[tf.newaxis, :]
            out = self.concatenate(
                [tf.broadcast_to(self.first_input, [inputs.shape[0], self.first_input.shape[1]]), inputs])
        else:
            out = self.first_input

        out = self.layers(out)

        logits = out[:, :self.num_components]
        means = out[:, self.num_components:2 * self.num_components]
        stds = self.stds(out[:, 2 * self.num_components:])

        return logits, means, stds

    def get_config(self):
        config = super(ParametrizeGaussianMixtureBlock, self).get_config()
        config.update({'depth': self.depth,
                       'width': self.width,
                       'num_components': self.num_components,
                       'conditioned': self.conditioned
                       })

        return config


class AutoregressiveFlow(tfk.Model):
    def __init__(self, num_components, parametrize_depth, parametrize_width):
        super(AutoregressiveFlow, self).__init__()

        self.num_components = num_components
        self.parametrize_depth = parametrize_depth
        self.parametrize_width = parametrize_width

        self.mixture_1_params = ParametrizeGaussianMixtureBlock(num_components=num_components,
                                                                width=self.parametrize_width,
                                                                depth=self.parametrize_depth,
                                                                conditioned=False)

        self.mixture_2_params = ParametrizeGaussianMixtureBlock(num_components=num_components,
                                                                width=self.parametrize_width,
                                                                depth=self.parametrize_depth,
                                                                conditioned=True)

        self.f_x = tfkl.Lambda(lambda x: tf.stack(x, axis=-1), name='stack_x1_x2')

        self.mesh_X, self.mesh_Y = tf.meshgrid(tf.linspace(-4., 4., 1000), tf.linspace(-4., 4., 1000))
        mesh = tf.stack([self.mesh_X, self.mesh_Y], axis=-1)
        self.mesh_flattened = tf.reshape(mesh, (-1, 2))

        grid_vertical_X, grid_vertical_Y = tf.meshgrid(tf.linspace(-4., 4., 25), tf.linspace(-4., 4., 1000))
        grid_horizontal_X, grid_horizontal_Y = tf.meshgrid(tf.linspace(-4., 4., 1000),
                                                           tf.linspace(-4., 4., 25))
        self.gridlines = tf.concat([tf.transpose(tf.stack([grid_vertical_X, grid_vertical_Y], axis=-1), [1, 0, 2]),
                                    tf.stack([grid_horizontal_X, grid_horizontal_Y], axis=-1)],
                                   axis=0)

    def _get_distributions(self, inputs):
        x1 = inputs[:, 0:1]

        logits_1, means_1, stds_1 = self.mixture_1_params(None)
        logits_2, means_2, stds_2 = self.mixture_2_params(x1)

        mixture_1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logits_1),
                                          components_distribution=tfd.Normal(loc=means_1, scale=stds_1))

        mixture_2 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logits_2),
                                          components_distribution=tfd.Normal(loc=means_2, scale=stds_2))

        return mixture_1, mixture_2

    def _batched_function(self, data, function, batch_size):
        total_steps = np.ceil(len(data) / batch_size).astype(int)
        results = []
        for step in range(total_steps):
            indices = slice(step * batch_size, (step + 1) * batch_size)
            results.append(function(data[indices]))

        return tf.concat(results, axis=0)

    def transform(self, inputs):
        x1 = inputs[:, 0]
        x2 = inputs[:, 1]

        mixture_1, mixture_2 = self._get_distributions(inputs)

        return self.f_x([mixture_1.cdf(x1), mixture_2.cdf(x2)])

    def log_prob(self, inputs):
        x1 = inputs[:, 0]
        x2 = inputs[:, 1]

        mixture_1, mixture_2 = self._get_distributions(inputs)
        return mixture_1.log_prob(x1) + mixture_2.log_prob(x2)

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

        steps_per_epoch = np.ceil(len(X) / train_batch_size).astype(int)
        global_step = 0

        nll_loss = lambda X: tf.reduce_mean(-self.log_prob(X))

        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                random_indices = np.random.choice(len(X), size=train_batch_size)
                random_samples = X[random_indices]

                with tf.GradientTape() as tape:
                    train_loss = nll_loss(random_samples)

                with writer_train.as_default():
                    tf.summary.scalar('loss', train_loss, step=global_step)

                grads = tape.gradient(train_loss, self.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))

                if global_step % val_freq == 0:
                    with writer_val.as_default():
                        val_loss = self._batched_function(val_X,
                                                          function=nll_loss,
                                                          batch_size=eval_batch_size)
                        tf.summary.scalar('loss', val_loss, step=global_step)

                if global_step % plot_freq == 0:
                    density = self.density_plot(eval_batch_size=eval_batch_size)
                    latents = self.latents_plot(eval_batch_size=eval_batch_size, X=X, Y=Y)
                    grid_plot = self.transformed_grid_plot(eval_batch_size=eval_batch_size)

                    together = np.concatenate([density, latents, grid_plot], axis=1)
                    with writer_images.as_default():
                        tf.summary.image('plots', together[np.newaxis, :, :, :], step=global_step)

                global_step += 1


def __main__():
    gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, enable=True)

    X, Y = sample_data()
    train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size=0.2)

    autoregressive_flow = AutoregressiveFlow(num_components=10,
                                             parametrize_depth=10,
                                             parametrize_width=128)

    adam = tfk.optimizers.Adam(1e-3)
    autoregressive_flow.fit(X=train_X, Y=train_Y, val_X=val_X,
                            epochs=1000,
                            optimizer=adam,
                            logdir=f'logs/hw_2_1_1/{generate_slug()}',
                            plot_freq=5,
                            train_batch_size=1024,
                            val_freq=5,
                            eval_batch_size=32768)


if __name__ == '__main__':
    __main__()
