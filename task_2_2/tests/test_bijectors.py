import unittest
import tensorflow_probability as tfp
import numpy as np
from task_2_2.bijectors import *


class TestSplits(unittest.TestCase):
    def test_checkerboard(self):
        img = np.array(
            [[[[1, 2], [3, 4]],
              [[5, 6], [7, 8]]],
             [[[9, 10], [11, 12]],
              [[13, 14], [15, 16]]]],
            dtype=np.float32
        )

        checkerboard_split = CheckerboardSplit()
        result = checkerboard_split(img)
        x1, x2 = tf.split(result, 2, axis=-1)

        np.testing.assert_array_equal(x1, np.array([[[[1, 2]],
                                                     [[7, 8]]],
                                                    [[[9, 10]],
                                                     [[15, 16]]]]).astype(np.float32))

        np.testing.assert_array_equal(x2, np.array([[[[3, 4]],
                                                     [[5, 6]]],
                                                    [[[11, 12]],
                                                     [[13, 14]]]]).astype(np.float32))

        chain = tfp.bijectors.Chain([CheckerboardSplit(), tfp.bijectors.Invert(CheckerboardSplit())])
        np.testing.assert_array_equal(chain(img), img)
        np.testing.assert_array_equal(chain.inverse_log_det_jacobian(img, event_ndims=4), [0., 0.])


    def test_squeeze(self):
        img = np.random.randn(10, 32, 32, 3)
        squeeze_and_back = Squeeze()
        squeeze_and_back = tfb.Invert(Squeeze())(squeeze_and_back)
        np.testing.assert_array_equal(img, squeeze_and_back(img))