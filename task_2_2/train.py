import tensorflow as tf
import pickle
import math
import argparse
import json
from coolname import generate_slug
from task_2_2.utils import preprocess, postprocess
from task_2_2.chains import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--restore', type=str, help='restore weights from the checkpoint')
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--warmup_steps', type=int, help='number of update steps to increase learning rate from 0 '
                                                     'to target learning rate', default=0)
parser.add_argument('--anneal_steps', type=int, default=0)
parser.add_argument('--total_steps', type=int, required=True)
parser.add_argument('--plot_samples_freq', type=int, help='the frequency of plotting samples in steps')
parser.add_argument('--validate_freq', type=int, help='the frequency of calculating loss on the whole validation '
                                                      'set in steps',
                    default=1000)
parser.add_argument('--save_freq', type=int, help='the frequency of saving weights in steps', default=500)
parser.add_argument('--log_histograms', action='store_true', help='log histograms of activations and '
                                                                  'some parameters every training step')
parser.add_argument('--fp16', action='store_true', help='use automated mixed precision')
parser.add_argument('--chain', choices=['real_nvp', 'multiscale_real_nvp', 'multiscale_glow'],
                    help='the type of the Flow model',
                    required=True)
parser.add_argument('--filters', type=int,
                    help='the number of filters in each convolution of affine coupling shift and scale mapping',
                    default=256)
parser.add_argument('--blocks', type=int, help='the number of resnet blocks in affine coupling shift and scale mapping',
                    default=6)
parser.add_argument('--steps_per_scale', type=int, help='')
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()

tfk = tf.keras
tfkl = tf.keras.layers

tfd = tfp.distributions
tfb = tfp.bijectors

with open(args.dataset, 'rb') as fp:
    data = pickle.load(fp)




gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, enable=True)

batch_size = 64
shape = [32, 32, 3]
n_bits = 2.

train_X = tf.data.Dataset.from_tensor_slices(data['train']) \
    .map(lambda X: preprocess(X, n_bits), num_parallel_calls=8).shuffle(1000).repeat().batch(batch_size).prefetch(1)
val_X = tf.data.Dataset.from_tensor_slices(data['test']) \
    .map(lambda X: preprocess(X, n_bits), num_parallel_calls=8).repeat().batch(batch_size).prefetch(1)

val_steps_per_epoch = math.ceil(len(data['test']) / batch_size)

make_chain = {
    'real_nvp': lambda: real_nvp(shape, filters=args.filters, blocks=args.blocks),
    'multiscale_real_nvp': lambda: multiscale_real_nvp(shape,
                                                       filters=args.filters,
                                                       blocks=args.blocks),
    'multiscale_glow': lambda: multiscale_glow(shape, steps_per_scale=args.steps_per_scale, filters=args.filters)
}

with tf.device('GPU:0'):
    chain = make_chain[args.chain]()
    transformed_distribution = tfd.TransformedDistribution(
        event_shape=[tf.reduce_prod(shape)],
        distribution=tfd.Normal(loc=0.0, scale=1.0),
        bijector=tfb.Invert(chain)
    )

slug = generate_slug()
train_writer = tf.summary.create_file_writer(f'logs/{slug}/train')
val_writer = tf.summary.create_file_writer(f'logs/{slug}/val')
with open(f'logs/{slug}/args.json', 'w') as fp:
    json.dump(vars(args), fp, indent=4)

learning_rate = tf.Variable(args.learning_rate, trainable=False)
optimizer = tf.optimizers.Adam(learning_rate)
if args.fp16:
    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)


@tf.function
def sample(rows, columns):
    samples = transformed_distribution.sample(rows * columns)
    samples = postprocess(samples, n_bits)
    samples = samples[:, :, :, ::-1]
    samples = tf.reshape(samples, (rows, columns, *shape))
    samples = tf.transpose(samples, (0, 2, 1, 3, 4))
    samples = tf.reshape(samples, (1, shape[0] * rows, shape[1] * columns, shape[2]))

    return samples


@tf.function
def compute_and_minimize_loss(batch, minimize):
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

for batch in train_X:
    if step < args.warmup_steps:
        learning_rate.assign(
            args.learning_rate * tf.cast(step + 1, tf.float32) / tf.cast(args.warmup_steps, tf.float32))
    elif step > args.total_steps - args.anneal_steps and step < args.total_steps:
        anneal_step = step - args.total_steps + args.anneal_steps
        learning_rate.assign(tf.cast((1 - anneal_step / args.anneal_steps) * args.learning_rate, dtype=tf.float32))
    elif step == args.total_steps:
        break

    if args.log_histograms:
        with train_writer.as_default():
            loss = compute_and_minimize_loss(batch, minimize=True)
    else:
        loss = compute_and_minimize_loss(batch, minimize=True)

    with train_writer.as_default():
        tf.summary.scalar('lr', learning_rate)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('bits/dim', bits_dim(loss))

        if args.plot_samples_freq and step.numpy() % args.plot_samples_freq == 0:
            samples = sample(rows=4, columns=10)
            tf.summary.image('samples', samples)

    if step.numpy() % args.validate_freq == 0:
        losses = []
        bits_dims = []
        for i, batch in enumerate(val_X):
            losses.append(compute_and_minimize_loss(batch, minimize=False))
            bits_dims.append(bits_dim(losses[-1]))

            if i == val_steps_per_epoch - 1:
                break

        with val_writer.as_default():
            tf.summary.scalar('loss', tf.reduce_mean(losses))
            tf.summary.scalar('bits/dim', tf.reduce_mean(bits_dims))

    if step.numpy() % args.save_freq == 0 and step.numpy():
        variables = {}
        for v in transformed_distribution.trainable_variables:
            variables[v.name] = v.value().numpy()

        with open('logs/{}/weights_{:05d}.pickle'.format(slug, step.numpy()), 'wb') as fp:
            pickle.dump(variables, fp)

    step.assign(step + 1)

variables = {}
for v in transformed_distribution.trainable_variables:
    variables[v.name] = v.value().numpy()

with open('logs/{}/weights_{:05d}.pickle'.format(slug, step.numpy()), 'wb') as fp:
    pickle.dump(variables, fp)
