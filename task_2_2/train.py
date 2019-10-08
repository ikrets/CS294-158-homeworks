import tensorflow as tf
import tensorflow_probability as tfp
import pickle
import math
import argparse
import os
import traceback
from coolname import generate_slug
from task_2_2.chains import default_chain

parser = argparse.ArgumentParser()
parser.add_argument('--restore', type=str, help='restore weights from the checkpoint')
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--warmup_steps', type=int, help='number of update steps to increase learning rate from 0 '
                                                     'to target learning rate')
parser.add_argument('--plot_samples_freq', type=int, help='the frequency of plotting samples in steps')
parser.add_argument('--validate_freq', type=int, help='the frequency of calculating loss on the whole validation '
                                                      'set in steps')
args = parser.parse_args()

tfk = tf.keras
tfkl = tf.keras.layers

tfd = tfp.distributions
tfb = tfp.bijectors

with open('hw2_q2.pkl', 'rb') as fp:
    data = pickle.load(fp)


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


gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, enable=True)

batch_size = 64
shape = [32, 32, 3]
n_bits = 2.

train_X = tf.data.Dataset.from_tensor_slices(data['train']) \
    .map(lambda X: preprocess(X, n_bits), num_parallel_calls=8).shuffle(1000).batch(batch_size).repeat().prefetch(1)
val_X = tf.data.Dataset.from_tensor_slices(data['test']) \
    .map(lambda X: preprocess(X, n_bits), num_parallel_calls=8).batch(batch_size).repeat().prefetch(1)

val_steps_per_epoch = math.ceil(len(data['test']) / batch_size)

chain = default_chain(shape=shape, filters=256, blocks=6)
transformed_distribution = tfd.TransformedDistribution(
    event_shape=[tf.reduce_prod(shape)],
    distribution=tfd.Normal(loc=0.0, scale=1.0),
    bijector=tfb.Invert(chain)
)

slug = generate_slug()
train_writer = tf.summary.create_file_writer(f'logs/{slug}/train')
val_writer = tf.summary.create_file_writer(f'logs/{slug}/val')

learning_rate = tf.Variable(args.learning_rate, trainable=False)
optimizer = tf.optimizers.Adam(learning_rate)
optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)


@tf.function
def sample(rows, columns):
    samples = transformed_distribution.sample(rows * columns)
    mean_log_prob = tf.reduce_mean(transformed_distribution.log_prob(samples))

    samples = postprocess(samples, n_bits)
    samples = samples[:, :, :, ::-1]
    samples = tf.reshape(samples, (rows, columns, *shape))
    samples = tf.transpose(samples, (0, 2, 1, 3, 4))
    samples = tf.reshape(samples, (1, shape[0] * rows, shape[1] * columns, shape[2]))

    return samples, mean_log_prob

@tf.function
def compute_and_minimize_loss(batch, minimize=True):
    var_count = tf.cast(tf.reduce_prod(batch.shape[1:]), tf.float32)

    with tf.GradientTape() as tape:
        loss = -tf.reduce_mean(transformed_distribution.log_prob(batch)) + var_count * tf.math.log((2 ** n_bits) / .95)

    if minimize:
        grads = tape.gradient(loss, chain.trainable_variables)
        optimizer.apply_gradients(zip(grads, chain.trainable_variables))

    return loss


@tf.function
def bits_dim(loss):
    return loss / tf.math.log(2.) / tf.cast(tf.reduce_prod(shape), tf.float32)


step = tf.Variable(0, trainable=False, dtype=tf.int64)
tf.summary.experimental.set_step(step)

if args.restore:
    with open(args.restore, 'rb') as fp:
        print('Loading trainable variables...')
        weights = pickle.load(fp)
        for v in transformed_distribution.trainable_variables:
            v.assign(weights[v.name])

try:
    with train_writer.as_default():
        for batch in train_X:
            if step < args.warmup_steps:
                learning_rate.assign(
                    args.learning_rate * tf.cast(step + 1, tf.float32) / tf.cast(args.warmup_steps, tf.float32))

            tf.summary.scalar('lr', learning_rate)

            loss = compute_and_minimize_loss(batch, minimize=True)

            tf.summary.scalar('loss', loss)
            tf.summary.scalar('bits/dim', bits_dim(loss))

            if step.numpy() % args.plot_samples_freq == 0:
                samples, mean_log_prob = sample(rows=4, columns=10)
                tf.summary.image('samples', samples)
                tf.summary.scalar('samples_mean_log_prob', mean_log_prob)

            if step.numpy() % args.validate_freq == 0:
                with val_writer.as_default():
                    losses = []
                    bits_dims = []
                    for i, batch in enumerate(val_X):
                        losses.append(compute_and_minimize_loss(batch, minimize=False))
                        bits_dims.append(bits_dim(losses[-1]))

                        if i == val_steps_per_epoch - 1:
                            break
                    tf.summary.scalar('loss', tf.reduce_mean(losses))
                    tf.summary.scalar('bits/dim', tf.reduce_mean(bits_dims))

            step.assign(step + 1)
except KeyboardInterrupt:
    traceback.print_exc()

    print('Saving trainable variables...')
    variables = {}
    for v in transformed_distribution.trainable_variables:
        variables[v.name] = v.value().numpy()

    os.makedirs('weights', exist_ok=True)
    with open(f'weights/{slug}.pickle', 'wb') as fp:
        pickle.dump(variables, fp)
