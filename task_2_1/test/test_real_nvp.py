import unittest
from task_2_1.real_nvp import AffineCouplingLayer, LastChannelSplit
import tensorflow as tf
import numpy as np

tfk = tf.keras
tfkl = tf.keras.layers


class AffineCouplingTest(unittest.TestCase):
    def test_coupling(self):
        data = np.random.randn(3, 2).astype(np.float32)

        const_s = lambda X: tf.concat([tf.ones_like(X), tf.zeros_like(X)], axis=-1)
        const_t = lambda X: tf.concat([tf.ones_like(X), tf.ones_like(X)], axis=-1)

        part1_coupling = AffineCouplingLayer(LastChannelSplit(flip=False), const_s)
        part2_coupling = AffineCouplingLayer(LastChannelSplit(flip=True), const_s)

        split = np.array([1, 0])
        sigmoid_1 = np.exp(1) / (1 + np.exp(1))

        np.testing.assert_array_equal(data * ((1 - split) * sigmoid_1 + split * tf.ones_like(data)), part1_coupling(data))
        np.testing.assert_array_equal(data * (split * sigmoid_1 + (1 - split) * tf.ones_like(data)), part2_coupling(data))

        np.testing.assert_array_almost_equal(tf.ones(3) * np.log(sigmoid_1), part1_coupling.log_determinant(data))
        np.testing.assert_array_almost_equal(tf.ones(3) * np.log(sigmoid_1), part2_coupling.log_determinant(data))

        part1_coupling = AffineCouplingLayer(LastChannelSplit(flip=False), const_t)
        part2_coupling = AffineCouplingLayer(LastChannelSplit(flip=True), const_t)

        np.testing.assert_array_equal(data * ((1 - split) * sigmoid_1 + split * tf.ones_like(data)) + (1 - split) * 1, part1_coupling(data))
        np.testing.assert_array_equal(data * (split * sigmoid_1 + (1 - split) * tf.ones_like(data)) + split * 1, part2_coupling(data))

    def test_last_channel_split(self):
        split = LastChannelSplit(flip=False)
        p1, p2 = split.split(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
        np.testing.assert_array_equal([[1, 2], [5, 6]], p1)
        np.testing.assert_array_equal([[3, 4], [7, 8]], p2)

        random_img = np.random.randn(10, 8, 8, 4)
        p1, p2 = split.split(random_img)
        np.testing.assert_array_equal(random_img[..., :2], p1)
        np.testing.assert_array_equal(random_img[..., 2:], p2)


