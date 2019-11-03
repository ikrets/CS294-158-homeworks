import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import argparse
import coolname
import os

tfk = tf.keras
tfkl = tf.keras.layers
tfd = tfp.distributions
tfpl = tfp.layers


def canvas_to_numpy(fig):
    fig.canvas.draw()
    cols, rows = fig.canvas.get_width_height()
    plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((1, rows, cols, 3))
    plt.close()

    return plot


def sample_data_1(count=100000):
    rand = np.random.RandomState(0)
    return ([[1.0, 2.0]] + rand.randn(count, 2) * [[5.0, 1.0]]).astype(np.float32), None


def sample_data_2(count=100000):
    rand = np.random.RandomState(0)
    return [[1.0, 2.0]] + (rand.randn(count, 2) * [[5.0, 1.0]]).dot(
        [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2]]
    ).astype(np.float32), None


def sample_data_3(count=100000):
    rand = np.random.RandomState(0)
    a = [[-1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    b = [[1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    c = np.c_[2 * np.cos(np.linspace(0, np.pi, count // 3)), -np.sin(np.linspace(0, np.pi, count // 3))]
    c += rand.randn(*c.shape) * 0.2

    data_x = np.concatenate([a, b, c], axis=0)
    data_y = np.array([0] * len(a) + [1] * len(b) + [2] * len(c))
    perm = rand.permutation(len(data_x))

    return data_x[perm], data_y[perm]


def dense_parametrize(width, depth, final_shape):
    parametrization = [tfkl.Dense(width, activation='relu', name=f'parametrize_{i}') for i in range(depth - 1)]
    parametrization.append(tfkl.Dense(final_shape, name=f'parametrize_{depth - 1}'))
    return parametrization


def log_kl_divergence(kl_divergence):
    def kl_with_log(distribution):
        value = kl_divergence(distribution)
        tf.summary.scalar('kl_divergence', value)
        return value

    return kl_with_log


def encoder_1(width, depth, z_dimensions):
    encoder = dense_parametrize(width, depth,
                                tfpl.IndependentNormal.params_size([z_dimensions]))
    encoder.append(tfpl.IndependentNormal([z_dimensions]))
    return tfk.Sequential(encoder, name='encoder')


def decoder_1(width, depth, input_dimensions):
    decoder = dense_parametrize(width, depth,
                                tfpl.IndependentNormal.params_size([input_dimensions]))
    decoder.append(tfpl.IndependentNormal([input_dimensions]))
    return tfk.Sequential(decoder, name='decoder')


def decoder_2(width, depth, input_dimensions):
    decoder = dense_parametrize(width, depth,
                                input_dimensions + 1)
    decoder.append(tfpl.DistributionLambda(
        make_distribution_fn=lambda t: tfd.MultivariateNormalDiag(loc=t[..., :-1],
                                                                  scale_diag=(
                                                                      tf.math.softplus(
                                                                          tf.stack([t[..., -1], t[..., -1]],
                                                                                   axis=-1))))
    ))
    return tfk.Sequential(decoder, name='decoder')


def decoder_3(width, depth, input_dimensions):
    decoder = dense_parametrize(width, depth,
                                tfpl.MultivariateNormalTriL.params_size(input_dimensions))
    decoder.append(tfpl.MultivariateNormalTriL(input_dimensions))

    return tfk.Sequential(decoder, name='decoder')


def plot_distribution(d1, d2, distribution, dimension_name):
    means = distribution.mean()
    stds = distribution.stddev()

    fig, axes = plt.subplots(2, means.shape[-1], figsize=(15, 4 * means.shape[-1]))

    for i in range(means.shape[-1]):
        im = axes[0, i].pcolormesh(d1, d2, means[..., i])
        axes[0, i].set_xlabel(f'{dimension_name}_1')
        axes[0, i].set_ylabel(f'{dimension_name}_2')
        axes[0, i].set_title(f'$\mu_{i + 1}$')
        plt.colorbar(im, ax=axes[0, i])

        im = axes[1, i].pcolormesh(d1, d2, stds[..., i])
        axes[1, i].set_xlabel(f'{dimension_name}_1')
        axes[1, i].set_ylabel(f'{dimension_name}_2')
        axes[1, i].set_title(f'$\sigma_{i + 1}$')
        plt.colorbar(im, ax=axes[1, i])

    plt.tight_layout()
    return canvas_to_numpy(fig)


def plot_samples_and_means(x_distribution):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    samples = x_distribution.sample()
    means = x_distribution.mean()

    axes[0].scatter(samples[:, 0], samples[:, 1], alpha=0.01)
    axes[0].set_title('Samples')

    axes[1].scatter(means[:, 0], means[:, 1], alpha=0.01)
    axes[1].set_title('Means')

    plt.tight_layout()
    return canvas_to_numpy(fig)


def plot_labeled_latents(z_distribution, labels):
    fig = plt.figure(figsize=(4, 4))
    samples = z_distribution.sample()
    plt.scatter(samples[:, 0], samples[:, 1], c=labels)
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    plt.tight_layout()
    return canvas_to_numpy(fig)


def train_vae(prior, encoder, decoder, optimizer,
              data_x,
              batch_size,
              val_split,
              steps,
              detailed_log_freq,
              logdir,
              plot_x_grid,
              plot_z_grid,
              data_y=None
              ):
    trainable_variables = None
    train_writer = tf.summary.create_file_writer(os.path.join(logdir, 'train'))
    val_writer = tf.summary.create_file_writer(os.path.join(logdir, 'val'))

    ind_train = np.arange(int(len(data_x) * (1 - val_split)))
    ind_val = np.arange(int(len(data_x) * (1 - val_split)), len(data_x))

    x_train = data_x[ind_train]
    x_val = data_x[ind_val]

    if data_y is not None:
        y_train = data_y[ind_train]

    for step in range(steps):
        tf.summary.experimental.set_step(step)
        x = x_train[np.random.choice(len(x_train), size=batch_size)]

        with tf.GradientTape() as tape:
            q_z_x = encoder(x)
            kld = tf.reduce_mean(tfd.kl_divergence(q_z_x, prior))
            rv_x = decoder(q_z_x.sample())
            nll = tf.reduce_mean(-rv_x.log_prob(x))

            elbo = nll + kld

        if step % detailed_log_freq == 0:
            with train_writer.as_default():
                grid_z_x = encoder(plot_x_grid)
                plot = plot_distribution(plot_x_grid[..., 0], plot_x_grid[..., 1], distribution=grid_z_x,
                                         dimension_name='x')
                tf.summary.image('Q(z|x)', plot)

                grid_x_z = decoder(plot_z_grid)
                plot = plot_distribution(plot_z_grid[..., 0], plot_z_grid[..., 1], distribution=grid_x_z,
                                         dimension_name='z')
                tf.summary.image('P(x|z)', plot)

                samples = prior.sample(100000)
                rv_x = decoder(samples)
                plot = plot_samples_and_means(rv_x)
                tf.summary.image('Sampled X', plot)

        with train_writer.as_default():
            tf.summary.scalar('KL divergence', kld)
            tf.summary.scalar('Negative log likelihood', nll)
            tf.summary.scalar('ELBO', elbo)

        if trainable_variables is None:
            trainable_variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(elbo, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        if step % detailed_log_freq == 0:
            with val_writer.as_default():
                q_z_x = encoder(x_val)
                kld = tf.reduce_mean(tfd.kl_divergence(q_z_x, prior))
                rv_x = decoder(q_z_x.sample())
                nll = tf.reduce_mean(-rv_x.log_prob(x_val))
                elbo = nll + kld

                tf.summary.scalar('KL divergence', kld)
                tf.summary.scalar('Negative log likelihood', nll)
                tf.summary.scalar('ELBO', elbo)

            if data_y is not None:
                with train_writer.as_default():
                    q_z_x = encoder(x_train)
                    tf.summary.image('labeled latents', plot_labeled_latents(q_z_x, y_train))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['1.1', '1.2', '2'], required=True)
    parser.add_argument('--data', type=int, choices=[1, 2, 3], required=True)
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()

    z_dimensions = 2

    slug = coolname.generate_slug()
    prior = tfd.Independent(
        tfd.Normal(loc=tf.zeros(z_dimensions, dtype=tf.float32), scale=tf.ones(z_dimensions, dtype=tf.float32)),
        reinterpreted_batch_ndims=1)
    enc = encoder_1(1000, 3, z_dimensions=z_dimensions)
    dec = decoder_1(1000, 3, input_dimensions=2) if args.task != '1.2' else decoder_2(1000, 3, input_dimensions=2)

    adam = tf.keras.optimizers.Adam(1e-4)
    adam = tf.train.experimental.enable_mixed_precision_graph_rewrite(adam)

    if args.data == 1:
        data_x, data_y = sample_data_1(100000)
        data_x /= 40
    if args.data == 2:
        data_x, data_y = sample_data_2(100000)
        data_x /= 40
    if args.data == 3:
        data_x, data_y = sample_data_3(100000)
        data_x /= 8

    grid_z1, grid_z2 = tf.meshgrid(tf.linspace(-2., 2., 100), tf.linspace(-2., 2., 100))
    grid_z = tf.stack([grid_z1, grid_z2], axis=-1)

    space = tf.linspace(-15., 20., 100) / 40 if args.task != '2' else tf.linspace(-4., 4., 100) / 8
    grid_x1, grid_x2 = tf.meshgrid(space, space)
    grid_x = tf.stack([grid_x1, grid_x2], axis=-1)

    train_vae(prior, enc, dec, optimizer=adam,
              data_x=data_x,
              data_y=data_y,
              batch_size=1024,
              val_split=0.8,
              steps=args.steps,
              detailed_log_freq=500,
              logdir=f'logs/task_{args.task}_data_{args.data}_{slug}',
              plot_z_grid=grid_z,
              plot_x_grid=grid_x)
