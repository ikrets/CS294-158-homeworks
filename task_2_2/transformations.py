import tensorflow as tf

tfk = tf.keras
tfkl = tf.keras.layers

def simple_resnet(filters, blocks, channels, name='simple_resnet'):
    input = tfkl.Input(shape=[None, None, channels])
    net = tfkl.Conv2D(filters=filters, kernel_size=3, padding='same', use_bias=False, name=f'{name}/initial_conv')(
        input)
    net = tfkl.BatchNormalization()(net)
    net = tfkl.Activation('relu')(net)

    for i in range(blocks):
        block_input = net

        net = tfkl.Conv2D(filters=filters, kernel_size=1, use_bias=False, name=f'{name}/block_{i}_conv_0')(net)
        net = tfkl.BatchNormalization()(net)
        net = tfkl.Activation('relu')(net)

        net = tfkl.Conv2D(filters=filters, kernel_size=3, use_bias=False, padding='same',
                          name=f'{name}/block_{i}_conv_1')(net)
        net = tfkl.BatchNormalization()(net)
        net = tfkl.Activation('relu')(net)

        net = tfkl.Conv2D(filters=filters, kernel_size=1, use_bias=False, name=f'{name}/block_{i}_conv_2')(net)
        net = tfkl.BatchNormalization()(net)
        net = tfkl.Add()([block_input, net])
        net = tfkl.Activation('relu')(net)

    net = tfkl.Conv2D(filters=channels * 2, kernel_size=3, padding='same', name=f'{name}/final_conv')(net)
    net = tfkl.Lambda(lambda X: tf.split(X, 2, axis=-1))(net)

    return tfk.Model(inputs=input, outputs=net)
