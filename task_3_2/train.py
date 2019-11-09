import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import argparse
import pickle
import json
from coolname import generate_slug

import task_3_2.model as model
from task_3_2.pixel_cnn import PixelCNNPrior

tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], enable=True)

tfk = tf.keras
tfkl = tf.keras.layers

tfd = tfp.distributions


def bits_per_dim(log_prob, shape):
    return log_prob * np.log2(np.e) / shape[0] / shape[1] / shape[2]


def pack_image_batch(x, rows, columns):
    x = tf.reshape(x, [rows, columns, *x.shape[1:]])
    x = tf.transpose(x, [0, 2, 1, 3, 4])
    x = tf.reshape(x, [1, x.shape[0] * x.shape[1], x.shape[2] * x.shape[3],
                       x.shape[4]])
    return x


def train_vae(prior, encoder, decoder, optimizer,
              warmup_steps,
              epochs,
              steps_per_epoch,
              val_steps,
              data_train,
              data_val,
              val_freq,
              samples_freq,
              sample_row_cols,
              logdir):
    trainable_variables = None
    train_writer = tf.summary.create_file_writer(os.path.join(logdir, 'train'))
    val_writer = tf.summary.create_file_writer(os.path.join(logdir, 'val'))

    @tf.function
    def calc_losses(x):
        q_z_x = encoder(x)
        z_x = q_z_x.sample()
        kld = tf.reduce_mean(q_z_x.log_prob(z_x) - prior.log_prob(z_x))
        rv_x = decoder(z_x)

        nll = tf.reduce_mean(-rv_x.log_prob(x))

        return {'kld': kld, 'nll': nll, 'elbo': nll + kld}

    lr = optimizer.learning_rate.numpy()

    tf.summary.experimental.set_step(0)
    for epoch in range(epochs):
        for step, x in enumerate(data_train):
            current_step = tf.summary.experimental.get_step()
            tf.summary.experimental.set_step(current_step + 1)
            current_step += 1

            if current_step <= warmup_steps:
                optimizer.learning_rate = current_step / warmup_steps * lr

            with tf.GradientTape() as tape:
                losses = calc_losses(x)

            with train_writer.as_default():
                tf.summary.scalar('KL divergence', losses['kld'])
                tf.summary.scalar('Negative log likelihood', losses['nll'])
                tf.summary.scalar('ELBO', losses['elbo'])

            if trainable_variables is None:
                trainable_variables = encoder.trainable_variables + decoder.trainable_variables

                if hasattr(prior, 'trainable_variables'):
                    trainable_variables += prior.trainable_variables

            gradients = tape.gradient(losses['elbo'], trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))

            if step == steps_per_epoch - 1:
                break

        if epoch % samples_freq == 0:
            num_x = 10
            num_reconstructions = 9

            q_z_x = encoder(x[:num_x])
            z_x = q_z_x.sample(num_reconstructions)
            z_x = tf.reshape(z_x, [num_x * num_reconstructions, *z_x.shape[2:]])

            random_channel = np.random.choice(z_x.shape[-1])

            x_z = decoder(z_x)
            reconstructed_x_means = x_z.mean()
            reconstructed_x_means = tf.reshape(reconstructed_x_means,
                                               [num_reconstructions, num_x, *reconstructed_x_means.shape[1:]])
            reconstructed_x_means = tf.transpose(reconstructed_x_means, [1, 0, 2, 3, 4])
            x_and_reconstructions = tf.concat([x[:num_x, tf.newaxis, ...], reconstructed_x_means],
                                              axis=1)
            x_and_reconstructions = tf.reshape(x_and_reconstructions,
                                               [x_and_reconstructions.shape[0] * x_and_reconstructions.shape[1],
                                                *x_and_reconstructions.shape[2:]])

            z = prior.sample(sample_row_cols[0] * sample_row_cols[1])
            rv_x = decoder(z)
            x_means = rv_x.mean()

            with train_writer.as_default():
                tf.summary.image('samples',
                                 pack_image_batch(tf.clip_by_value(x_means / 2 + 0.5, 0., 1.), *sample_row_cols))
                tf.summary.image('reconstructions',
                                 pack_image_batch(tf.clip_by_value(x_and_reconstructions / 2 + 0.5, 0., 1.),
                                                  num_x, num_reconstructions + 1))
                tf.summary.image('q(z|x) random channel', pack_image_batch(z_x[..., random_channel:random_channel + 1],
                                                                           num_x,
                                                                           num_reconstructions))
                tf.summary.histogram('p(x|z) stddevs', x_z.stddev())
                tf.summary.histogram('q(z|x) stddevs', q_z_x.stddev())

        if epoch % val_freq == 0:
            loss_accumulator = {'kld': [], 'nll': [], 'elbo': []}
            for val_step, val_x in enumerate(data_val):
                losses = calc_losses(val_x)
                for k in losses:
                    loss_accumulator[k].append(losses[k])

                if val_step == val_steps - 1:
                    break

            with val_writer.as_default():
                tf.summary.scalar('KL divergence', np.mean(loss_accumulator['kld']))
                tf.summary.scalar('Negative log likelihood',
                                  np.mean(loss_accumulator['nll']))
                tf.summary.scalar('ELBO', np.mean(loss_accumulator['elbo']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--validation_freq', type=int, required=True)
    parser.add_argument('--samples_freq', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--fixed_std', type=float)
    parser.add_argument('--prior', choices=['std', 'pixel_cnn'], required=True)
    args = parser.parse_args()

    num_parallel_calls = 8

    with open(args.dataset, 'rb') as fp:
        dataset = pickle.load(fp)

    calc_steps = lambda data, batch_size: np.ceil(len(data) / batch_size)

    data_train = tf.data.Dataset.from_tensor_slices(dataset['train']). \
        map(lambda X: tf.cast(X, tf.float32) / 255 * 2 - 1, num_parallel_calls). \
        shuffle(1000). \
        repeat(). \
        batch(args.batch_size). \
        prefetch(tf.data.experimental.AUTOTUNE)

    data_val = tf.data.Dataset.from_tensor_slices(dataset['valid']). \
        map(lambda X: tf.cast(X, tf.float32) / 255 * 2 - 1, num_parallel_calls). \
        shuffle(1000). \
        repeat(). \
        batch(args.batch_size). \
        prefetch(tf.data.experimental.AUTOTUNE)

    slug = generate_slug()
    logdir = os.path.join('logs', slug)
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, 'parameters.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)

    input_encoder = tfkl.Input(dataset['train'].shape[1:])
    encoder = tfk.Model(inputs=input_encoder, outputs=model.encoder(input_encoder))

    input_decoder = tfkl.Input(encoder.output.shape[1:])
    decoder = tfk.Model(inputs=input_decoder, outputs=model.decoder(input_decoder, fixed_std=args.fixed_std))

    if args.prior == 'std':
        prior = tfd.Independent(distribution=tfd.Normal(loc=tf.zeros(encoder.output.shape[1:]), scale=1.),
                                reinterpreted_batch_ndims=3)
    if args.prior == 'pixel_cnn':
        prior = PixelCNNPrior(shape=encoder.output.shape[1:])

    optimizer = tfk.optimizers.Adam(2e-4)
    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    train_vae(prior, encoder, decoder, optimizer,
              warmup_steps=100,
              epochs=args.epochs,
              steps_per_epoch=calc_steps(dataset['train'], args.batch_size),
              val_steps=calc_steps(dataset['valid'], args.batch_size),
              data_train=data_train,
              data_val=data_val,
              val_freq=args.validation_freq,
              samples_freq=args.samples_freq,
              sample_row_cols=[10, 10],
              logdir=logdir)

    tf.saved_model.save(encoder, os.path.join(logdir, 'encoder'))
    tf.saved_model.save(decoder, os.path.join(logdir, 'decoder'))
    if args.prior == 'pixel_cnn':
        tf.saved_model.save(prior, os.path.join(logdir, 'prior'))
