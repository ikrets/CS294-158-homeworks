import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfk = tf.keras
tfkl = tf.keras.layers


class CheckerboardSplit(tfb.Bijector):
    def __init__(self, validate_args=False, name='checkerboard_split'):
        super(CheckerboardSplit, self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=3,
            inverse_min_event_ndims=3,
            is_constant_jacobian=True,
            name=name
        )

    def _forward(self, x):
        y1_part1 = x[:, ::2, ::2, :]
        y1_part2 = x[:, 1::2, 1::2, :]

        y2_part1 = x[:, ::2, 1::2, :]
        y2_part2 = x[:, 1::2, ::2, :]

        result_shape = [-1, x.shape[1], x.shape[2] // 2, x.shape[3]]

        y1 = tf.reshape(tf.stack([y1_part1, y1_part2], axis=2), result_shape)
        y2 = tf.reshape(tf.stack([y2_part1, y2_part2], axis=2), result_shape)

        return tf.concat([y1, y2], axis=-1)

    def _inverse(self, y):
        x1, x2 = tf.split(y, 2, axis=-1)

        part_shape = [-1, y.shape[1] // 2, 2, y.shape[2], x1.shape[3]]

        x1 = tf.reshape(x1, part_shape)
        x2 = tf.reshape(x2, part_shape)

        x1_part1 = x1[:, :, 0, ...]
        x1_part2 = x1[:, :, 1, ...]
        x2_part1 = x2[:, :, 0, ...]
        x2_part2 = x2[:, :, 1, ...]

        x_part1 = tf.reshape(tf.stack([x1_part1, x2_part1], axis=3),
                             [-1, y.shape[1] // 2, y.shape[2] * 2, y.shape[3] // 2])
        x_part2 = tf.reshape(tf.stack([x2_part2, x1_part2], axis=3),
                             [-1, y.shape[1] // 2, y.shape[2] * 2, y.shape[3] // 2])

        x = tf.reshape(tf.stack([x_part1, x_part2], axis=2),
                       [-1, y.shape[1], y.shape[2] * 2, y.shape[3] // 2])

        return x

    def _inverse_log_det_jacobian(self, y):
        return tf.constant(0., dtype=y.dtype)

    def _forward_event_shape(self, input_shape):
        return [input_shape[0], input_shape[1] // 2, input_shape[2] * 2]

    def _inverse_event_shape(self, output_shape):
        return [output_shape[0], output_shape[1] * 2, output_shape[2] // 2]


class AffineCoupling(tfb.Bijector):
    def __init__(self, shift_scale, scale_activation_function, validate_args=False, name='affine_coupling'):
        super(AffineCoupling, self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=3,
            inverse_min_event_ndims=3,
            is_constant_jacobian=False,
            name=name
        )

        self.shift_scale = shift_scale
        self.scale_activation_function = scale_activation_function

    def _forward(self, x):
        x1, x2 = tf.split(x, 2, axis=-1)
        shift, scale = self.shift_scale(x2)
        tf.summary.histogram(f'shifts', shift)
        tf.summary.histogram(f'scales', scale)

        activation = tf.concat([self.scale_activation_function(scale) * (x1 + shift), x2], axis=-1)
        tf.summary.histogram(f'activations', activation)

        return activation

    def _inverse(self, y):
        y1, y2 = tf.split(y, 2, axis=-1)
        shift, scale = self.shift_scale(y2)

        return tf.concat([y1 / self.scale_activation_function(scale) - shift, y2], axis=-1)

    def _forward_log_det_jacobian(self, x):
        _, x2 = tf.split(x, 2, axis=-1)
        _, scale = self.shift_scale(x2)

        return tf.reduce_sum(tf.math.log(self.scale_activation_function(scale)), axis=tf.range(1, x.shape.rank))

    def _inverse_log_det_jacobian(self, y):
        _, y2 = tf.split(y, 2, axis=-1)
        _, scale = self.shift_scale(y2)

        return tf.reduce_sum(-tf.math.log(self.scale_activation_function(scale)),
                             axis=tf.range(1, y.shape.rank))


def tuple_flip_permutation(num_channels):
    return tf.concat([tf.range(num_channels // 2, num_channels), tf.range(num_channels // 2)], axis=0)


class ActNorm(tfb.Bijector):
    def __init__(self, num_channels, validate_args=False, name='act_norm'):
        super(ActNorm, self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=3,
            inverse_min_event_ndims=3,
            is_constant_jacobian=False,
            name=name
        )

        with tf.name_scope(self.name):
            self.s = tf.Variable(name='s', initial_value=tf.zeros(num_channels), trainable=True)
            self.b = tf.Variable(name='b', initial_value=tf.zeros(num_channels), trainable=True)

        self.num_channels = num_channels

    @tf.function
    def _init(self, x):
        if not tf.reduce_any(self.s != 0):
            per_channel_mean = tf.reduce_mean(x, axis=tf.range(x.shape.rank - 1))
            per_channel_std = tf.math.reduce_std(x - per_channel_mean[tf.newaxis, :], axis=tf.range(x.shape.rank - 1))

            self.b.assign(-per_channel_mean)
            self.s.assign(1. / per_channel_std)

    def _forward(self, x):
        self._init(x)

        activation = self.s * (x + self.b)

        tf.summary.histogram('activations', activation)
        tf.summary.histogram('variables/s', self.s.value())
        tf.summary.histogram('variables/b', self.b.value())

        return activation

    def _inverse(self, y):
        return y / self.s - self.b

    def _forward_log_det_jacobian(self, x):
        # forward seems to be always called first, but just to be sure
        self._init(x)

        area = tf.reduce_prod(x.shape[1:-1])
        det = tf.cast(area, tf.float32) * tf.reduce_sum(tf.math.log(tf.abs(self.s)))
        return tf.broadcast_to(det, x.shape[0:1])

    def _inverse_log_det_jacobian(self, y):
        area = tf.reduce_prod(y.shape[1:-1])
        det = tf.cast(area, tf.float32) * tf.reduce_sum(tf.math.log(tf.abs(1 / self.s)))
        return tf.broadcast_to(det, y.shape[0:1])


class Squeeze(tfb.Bijector):
    def __init__(self, validate_args=False, name='squeeze'):
        super(Squeeze, self).__init__(validate_args=validate_args,
                                      name=name,
                                      forward_min_event_ndims=3,
                                      inverse_min_event_ndims=3,
                                      is_constant_jacobian=True)

    def _forward(self, x):
        orig_x_shape = x.shape
        x = tf.reshape(x, [x.shape[0], x.shape[1] // 2, 2, x.shape[2] // 2, 2, x.shape[3]])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [orig_x_shape[0], orig_x_shape[1] // 2, orig_x_shape[2] // 2, 4 * orig_x_shape[3]])

        return x

    def _inverse(self, y):
        orig_y_shape = y.shape
        y = tf.reshape(y, [y.shape[0], y.shape[1], y.shape[2], 2, 2, y.shape[3] // 4])
        y = tf.transpose(y, [0, 1, 3, 2, 4, 5])
        y = tf.reshape(y, [orig_y_shape[0], orig_y_shape[1] * 2, orig_y_shape[2] * 2, orig_y_shape[3] // 4])

        return y

    def _forward_log_det_jacobian(self, x):
        return tf.constant(0., dtype=x.dtype)

    def _forward_event_shape(self, input_shape):
        return [input_shape[0] // 2, input_shape[1] // 2, input_shape[2] * 4]

    def _inverse_event_shape(self, output_shape):
        return [output_shape[0] * 2, output_shape[1] * 2, output_shape[2] // 4]


class InvertibleConvolution(tfb.Bijector):
    def __init__(self, num_channels, validate_args=False, name='inverible_convolution'):
        super(InvertibleConvolution, self).__init__(validate_args=validate_args,
                                                    name=name,
                                                    forward_min_event_ndims=3,
                                                    inverse_min_event_ndims=3,
                                                    is_constant_jacobian=False)

        Z = tf.random.normal([num_channels, num_channels])
        Q, _ = tf.linalg.qr(Z)

        self.W = tf.Variable(Q[tf.newaxis, tf.newaxis, ...], trainable=True, name=f'{self.name}/W')

    def _forward(self, x):
        return tf.nn.conv2d(x, filters=self.W, strides=1, padding='VALID')

    def _inverse(self, y):
        return tf.nn.conv2d(y, filters=tf.linalg.inv(self.W), strides=1, padding='VALID')

    def _forward_log_det_jacobian(self, x):
        _, logdet = tf.linalg.slogdet(self.W[0, 0])
        return x.shape[1] * x.shape[2] * logdet

    def _inverse_log_det_jacobian(self, y):
        _, logdet = tf.linalg.slogdet(tf.linalg.inv(self.W[0, 0]))
        return y.shape[1] * y.shape[2] * logdet
