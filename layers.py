from abc import abstractmethod

import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils


@tf.function
def erosion2d(x, st_element, strides, padding,
              rates=(1, 1, 1, 1), data_format='NHWC'):
    x = tf.nn.erosion2d(value=x, filters=st_element, strides=((1,) + strides + (1,)),
                        dilations=rates, padding=padding.upper(), data_format=data_format)
    return x


@tf.function
def dilation2d(x, st_element, strides, padding,
               rates=(1, 1, 1, 1), data_format='NHWC'):
    """
    Example:
        st_element = tf.constant([
        [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
        [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
        [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]
        ])
    """
    x = tf.nn.dilation2d(input=x, filters=st_element, strides=((1,) + strides + (1,)),
                         dilations=rates, padding=padding.upper(), data_format=data_format)
    return x


class MorphLayer(tf.keras.layers.Layer):
    """
        Basic class for morphological layers (grayscale)
        for now assuming channel last
    """

    def __init__(self, num_filters, kernel_size, strides,
                 padding, kernel_initializer,
                 kernel_constraint, channel_axis):
        super(MorphLayer, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)

        # for we are assuming channel last
        self.channel_axis = channel_axis

        self.kernel = None

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      constraint=self.kernel_constraint)

        # Be sure to call this at the end
        super(MorphLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=1)
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters,)

    @abstractmethod
    def call(self, x):
        pass

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer,
            "kernel_constraint": self.kernel_constraint,
            "channel_axis": self.channel_axis,
        })
        return config


class Erosion2D(MorphLayer):
    """
    Erosion 2D Layer
    for now assuming channel last
    """

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', kernel_initializer='glorot_uniform',
                 kernel_constraint=None, channel_axis=-1):
        super(Erosion2D, self).__init__(num_filters, kernel_size, strides,
                                        padding, kernel_initializer,
                                        kernel_constraint, channel_axis)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.channel_axis = -1

    def call(self, x):
        outputs = tf.keras.backend.placeholder()
        for i in range(self.num_filters):
            out = tf.keras.backend.sum(erosion2d(x, self.kernel[..., i],
                                                 self.strides, self.padding), keepdims=True, axis=-1)
            if i == 0:
                outputs = out
            else:
                outputs = tf.keras.backend.concatenate([outputs, out])

        return outputs


class Dilation2D(MorphLayer):
    """
    Dilation 2D Layer
    for now assuming channel last
    """

    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', kernel_initializer='glorot_uniform',
                 kernel_constraint=None, channel_axis=-1,
                 ):
        super(Dilation2D, self).__init__(num_filters, kernel_size, strides,
                                         padding, kernel_initializer,
                                         kernel_constraint, channel_axis)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.channel_axis = -1

    def call(self, x):

        outputs = tf.keras.backend.placeholder()
        for i in range(self.num_filters):
            out = tf.keras.backend.sum(dilation2d(x, self.kernel[..., i],
                                                  self.strides, self.padding), keepdims=True, axis=-1)

            if i == 0:
                outputs = out
            else:
                outputs = tf.keras.backend.concatenate([outputs, out])

        return outputs


class SpatialMorph(tf.keras.layers.Layer):
    def __init__(self, num_filters):
        super(SpatialMorph, self).__init__()
        self.num_filters = num_filters

        self.conv2d_1 = tf.keras.layers.Conv2D(filters=self.num_filters, kernel_size=(3, 3), padding='same',
                                               bias_initializer='zeros')
        self.erosion2d_1 = Erosion2D(num_filters, kernel_size=(3, 3), padding="same", strides=(1, 1),
                                     kernel_initializer=tf.keras.initializers.HeNormal())

        self.dilation2d_1 = Dilation2D(num_filters, kernel_size=(3, 3), padding="same", strides=(1, 1),
                                       kernel_initializer=tf.keras.initializers.HeNormal())
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=self.num_filters, kernel_size=(3, 3), padding='same',
                                               bias_initializer='zeros')
        self.add_1 = tf.keras.layers.Add()
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.relu_1 = tf.keras.layers.ReLU()

    def call(self, x):
        z1 = self.erosion2d_1(x)
        z1 = self.conv2d_1(z1)
        z2 = self.dilation2d_1(x)
        z2 = self.conv2d_2(z2)
        x = self.add_1([z1, z2])
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_filters": self.num_filters
        })
        return config


class SpectralMorph(tf.keras.layers.Layer):
    def __init__(self, num_filters):
        super(SpectralMorph, self).__init__()
        self.num_filters = num_filters
        self.erosion2d_1 = Erosion2D(num_filters=self.num_filters, kernel_size=(3, 3), padding="same", strides=(1, 1),
                                     kernel_initializer=tf.keras.initializers.HeNormal()
                                     )
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=self.num_filters, kernel_size=(1, 1), padding='same',
                                               bias_initializer='zeros')
        self.dilation2d_1 = Dilation2D(num_filters=self.num_filters, kernel_size=(3, 3), padding="same", strides=(1, 1),
                                       kernel_initializer=tf.keras.initializers.HeNormal())
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=self.num_filters, kernel_size=(1, 1), padding='same',
                                               bias_initializer='zeros')
        self.add_1 = tf.keras.layers.Add()
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.relu_1 = tf.keras.layers.ReLU()

    def call(self, x):
        z1 = self.erosion2d_1(x)
        z1 = self.conv2d_1(z1)
        z2 = self.dilation2d_1(x)
        z2 = self.conv2d_2(z2)
        x = self.add_1([z1, z2])
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_filters": self.num_filters
        })
        return config
