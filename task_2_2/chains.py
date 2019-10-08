import tensorflow as tf
import tensorflow_probability as tfp
from task_2_2.bijectors import CheckerboardSplit, AffineCoupling, tuple_flip_permutation, ActNorm, Squeeze
from task_2_2.transformations import simple_resnet

tfb = tfp.bijectors

def default_chain(shape, filters, blocks):
    checkerboard = CheckerboardSplit()

    scale_activation_function = lambda x: tf.maximum(tf.math.sigmoid(x), 1e-6)

    chain = [CheckerboardSplit()]
    checkerboard_channels = checkerboard.forward_event_shape(shape)[-1]
    for i in range(4):
        if i:
            chain.append(ActNorm(num_channels=checkerboard_channels, name=f'stage_0_act_norm_{i}'))
        chain.append(
            AffineCoupling(scale_activation_function=scale_activation_function,
                           shift_scale=simple_resnet(filters=filters, blocks=blocks,
                                                     channels=checkerboard_channels // 2,
                                                     name=f'stage_0_affine_coupling_shift_scale_{i}')))
        chain.append(tfb.Permute(permutation=tuple_flip_permutation(checkerboard_channels)))
    chain.append(tfb.Invert(CheckerboardSplit()))

    chain.append(Squeeze())
    squeezed_channels = shape[2] * 4
    for i in range(3):
        chain.append(ActNorm(squeezed_channels, name=f'stage_1_act_norm_{i}'))
        chain.append(AffineCoupling(scale_activation_function=scale_activation_function,
                                    shift_scale=simple_resnet(filters=filters, blocks=blocks,
                                                              channels=squeezed_channels // 2,
                                                              name=f'stage_1_affine_coupling_shift_scale_{i}')))
        chain.append(tfb.Permute(permutation=tuple_flip_permutation(squeezed_channels)))

    chain.append(CheckerboardSplit())
    checkerboard_channels = checkerboard.forward_event_shape([shape[0] // 2, shape[1] // 2, shape[2] * 4])[-1]
    for i in range(3):
        chain.append(ActNorm(checkerboard_channels, name=f'stage_2_act_norm_{i}'))
        chain.append(AffineCoupling(scale_activation_function=scale_activation_function,
                                    shift_scale=simple_resnet(filters=filters, blocks=blocks,
                                                              channels=checkerboard_channels // 2,
                                                              name=f'stage_2_affine_coupling_shift_scale_{i}')))
        chain.append(tfb.Permute(permutation=tuple_flip_permutation(checkerboard_channels)))
    chain.append(tfb.Invert(CheckerboardSplit()))

    chain.append(Squeeze())
    squeezed_channels = shape[2] * 16
    for i in range(3):
        chain.append(ActNorm(squeezed_channels, name=f'stage_3_act_norm_{i}'))
        chain.append(
            AffineCoupling(scale_activation_function=scale_activation_function,
                           shift_scale=simple_resnet(filters=filters, blocks=blocks, channels=squeezed_channels // 2,
                                                     name=f'stage_3_affine_coupling_shift_scale_{i}')))
        chain.append(tfb.Permute(permutation=tuple_flip_permutation(squeezed_channels)))

    chain.append(CheckerboardSplit())
    checkerboard_channels = checkerboard.forward_event_shape([shape[0] // 4, shape[1] // 4, shape[2] * 16])[-1]
    for i in range(3):
        chain.append(ActNorm(checkerboard_channels, name=f'stage_4_act_norm_{i}'))
        chain.append(AffineCoupling(scale_activation_function=scale_activation_function,
                                    shift_scale=simple_resnet(filters=filters, blocks=blocks,
                                                              channels=checkerboard_channels // 2,
                                                              name=f'stage_4_affine_coupling_shift_scale_{i}')))
        chain.append(tfb.Permute(permutation=tuple_flip_permutation(checkerboard_channels)))

    chain.append(ActNorm(checkerboard_channels, name='final_act_norm'))
    chain.append(tfb.Invert(CheckerboardSplit()))

    chain.append(tfb.Reshape(event_shape_in=[shape[0] // 4, shape[1] // 4, shape[2] * 16],
                             event_shape_out=[shape[0] * shape[1] * shape[2]]))

    chain.reverse()
    return tfb.Chain(chain, name='default_chain')
