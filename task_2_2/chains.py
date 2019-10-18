import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from task_2_2.bijectors import CheckerboardSplit, AffineCoupling, tuple_flip_permutation, ActNorm, Squeeze
from task_2_2.transformations import simple_resnet

tfb = tfp.bijectors


def real_nvp(shape, filters, blocks):
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
    return tfb.Chain(chain, name='real_nvp')


def multiscale_real_nvp(shape, steps_per_scale, filters, blocks):
    checkerboard = CheckerboardSplit()
    squeeze = Squeeze()

    scale_activation_function = lambda x: tf.maximum(tf.sigmoid(x), 1e-6)

    current_shape = shape
    L = np.log2(shape[0] // 4).astype(int) + 1
    factored_out = 0

    chain = []

    for level in range(L):
        scale_chain = []

        if level:
            scale_chain.append(tfb.Reshape([current_shape[2], current_shape[0], current_shape[1]]))
            scale_chain.append(tfb.Transpose([1, 2, 0]))

        scale_chain.append(CheckerboardSplit())
        checkerboard_channels = checkerboard.forward_event_shape(current_shape)[-1]

        for i in range(3 if level != L - 1 else 4):
            scale_chain.append(
                ActNorm(num_channels=checkerboard_channels,
                        name=f'level_{level}/checkerboard_{i}/act_norm_{i}'))
            scale_chain.append(AffineCoupling(scale_activation_function=scale_activation_function,
                                              shift_scale=simple_resnet(filters, blocks,
                                                                        channels=checkerboard_channels // 2,
                                                                        name=f'level_{level}/checkerboard_{i}/shift_scale_{i}')))
            scale_chain.append(tfb.Permute(permutation=tuple_flip_permutation(num_channels=checkerboard_channels)))
        scale_chain.append(tfb.Invert(CheckerboardSplit()))

        if level != L - 1:
            scale_chain.append(Squeeze())
            current_shape = squeeze.forward_event_shape(current_shape)
            filters *= 2

            for i in range(3):
                scale_chain.append(
                    ActNorm(num_channels=current_shape[-1], name=f'level_{level}/channel_{i}/act_norm_{i}'))
                scale_chain.append(AffineCoupling(scale_activation_function=scale_activation_function,
                                                  shift_scale=simple_resnet(filters, blocks,
                                                                            channels=current_shape[-1] // 2,
                                                                            name=f'level_{level}/channel_{i}/shift_scale_{i}')))
                scale_chain.append(tfb.Permute(permutation=tuple_flip_permutation(num_channels=current_shape[-1])))

        total_dimensions = current_shape[0] * current_shape[1] * current_shape[2]
        scale_chain.append(ActNorm(num_channels=current_shape[-1], name=f'level_{level}/final_act_norm'))
        scale_chain.append(tfb.Transpose([2, 0, 1]))
        scale_chain.append(tfb.Reshape(event_shape_in=[current_shape[2], current_shape[0], current_shape[1]],
                                       event_shape_out=[total_dimensions]))

        scale_chain.reverse()
        scale_chain = tfb.Chain(scale_chain, name=f'level_{level}')

        if level:
            chain.append(tfb.Blockwise(bijectors=[tfb.Identity(), scale_chain],
                                       block_sizes=[factored_out + (total_dimensions if level != L - 1 else 0),
                                                    total_dimensions]))
            factored_out += total_dimensions
        else:
            chain.append(scale_chain)

        if level != L - 2:
            current_shape = [current_shape[0], current_shape[1], current_shape[2] // 2]

    chain.reverse()
    return tfb.Chain(chain, name='multiscale_real_nvp')
