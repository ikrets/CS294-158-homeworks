import tensorflow_probability as tfp
import pickle
import json
from pathlib import Path
from task_2_2.chains import *

tfd = tfp.distributions


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

def latest_weights(log_dir):
    weights = Path(log_dir).glob('weights_*.pickle')
    return max(map(str, weights))


def load_model(args_file, weights_file):
    with open(args_file, 'r') as fp:
        args = json.load(fp)

    if args['chain'] == 'real_nvp':
        chain = real_nvp([32, 32, 3], filters=args['filters'], blocks=args['blocks'])
    elif args['chain'] == 'multiscale_real_nvp':
        chain = multiscale_real_nvp([32, 32, 3], filters=args['filters'], blocks=args['blocks'])
    elif args['chain'] == 'multiscale_glow':
        chain = multiscale_glow([32, 32, 3], steps_per_scale=args['steps_per_scale'], filters=args['filters'])
    elif args['chain'] == 'glow_no_factoring':
        chain = glow_no_factoring([32, 32, 3], filters=args['filters'], blocks=args['blocks'])
    else:
        raise RuntimeError(f'{args["chain"]} is not a valid chain name!')

    transformed_distribution = tfd.TransformedDistribution(event_shape=[32 * 32 * 3],
                                                           distribution=tfd.Normal(loc=0.0, scale=1.0),
                                                           bijector=tfb.Invert(chain))

    with open(weights_file, 'rb') as fp:
        weights = pickle.load(fp)

    for v in transformed_distribution.trainable_variables:
        v.assign(weights[v.name])

    return transformed_distribution


def pack(images, rows, first_reshape=True):
    if first_reshape:
        images = images.reshape([rows, -1, *images.shape[1:]])
    images = np.transpose(images, [0, 2, 1, 3, 4])
    images = images.reshape([images.shape[0] * images.shape[1], images.shape[2] * images.shape[3], images.shape[4]])
    return images
