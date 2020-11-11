import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers


class Component:
    """
        This class offers network component implementations.
        @Author: chen hao
        @Date:   2020.5.7
    """

    @staticmethod
    def get_conv_block(in_dim, out_dim, k=4, s=2, norm=True, non_linear='leaky_relu'):
        """
        build and return the basic convolution blocks, please refer to page.10 or Fig.6 in paper.

        :param in_dim: input tensor channels (dimensions).
        :param out_dim: output tensor channels (dimensions).
        :param k: convolution kernel size on both directions.
        :param s: convolution stride
        :param norm: use instance norm or not
        :param non_linear: non-linear function type
        :raise ValueError
        :return: tensorflow.keras.model object
        """
        layer_stack = []

        # Input Layer
        layer_stack += [tf.keras.Input(shape=(None, None, in_dim))]

        # Convolution Layer
        layer_stack += [tf.keras.layers.Conv2D(out_dim, kernel_size=k, strides=s, padding='same')]
        # InstanceNormalization Layer
        if norm is True:
            layer_stack += [tfa.layers.normalizations.InstanceNormalization(axis=3,
                                                                            center=True,
                                                                            scale=True,
                                                                            beta_initializer="random_uniform",
                                                                            gamma_initializer="random_uniform"
                                                                            )]
        # Non-linearity Layer
        if non_linear == 'leaky_relu':
            layer_stack += [tf.keras.layers.LeakyReLU()]
        elif non_linear == 'relu':
            layer_stack += [tf.keras.layers.ReLU()]
        elif non_linear == 'none':
            pass
        else:
            raise ValueError('No such non-linear layer found, got %s' % non_linear)

        return tf.keras.Sequential(layer_stack)

    @staticmethod
    def get_deconv_block(in_dim, out_dim, k=4, s=2, norm=True, non_linear='leaky_relu'):
        """
                build and return the basic convolution blocks, please refer to page.10 or Fig.6 in paper.

                :param in_dim: input tensor channels (dimensions).
                :param out_dim: output tensor channels (dimensions).
                :param k: convolution kernel size on both directions.
                :param s: convolution stride
                :param norm: use instance norm or not
                :param non_linear: non-linear function type
                :raise ValueError
                :return: tensorflow.keras.model object
        """
        layer_stack = []

        # Input Layer
        layer_stack += [tf.keras.Input(shape=(None, None, in_dim))]

        # Deconvolution
        layer_stack += [tf.keras.layers.Conv2DTranspose(out_dim, kernel_size=k, strides=s, padding='same')]

        # InstanceNormalization Layer
        if norm is True:
            layer_stack += [tfa.layers.InstanceNormalization(axis=3,
                                                             center=True,
                                                             scale=True,
                                                             beta_initializer="random_uniform",
                                                             gamma_initializer="random_uniform")]
        # Non-linearity Layer
        if non_linear == 'leaky_relu':
            layer_stack += [tf.keras.layers.LeakyReLU()]
        elif non_linear == 'relu':
            layer_stack += [tf.keras.layers.ReLU()]
        elif non_linear == 'tanh':
            layer_stack += [tf.keras.layers.Activation('tanh')]
        elif non_linear == 'none':
            pass
        else:
            raise ValueError('No such non-linear layer found, got %s' % non_linear)

        return tf.keras.Sequential(layer_stack)

    @staticmethod
    def get_res_block(in_dim, out_dim):
        """
        build and return the basic convolution blocks, please refer to page.10 or Fig.6 in paper.
        :param in_dim: input tensor channels (dimensions).
        :param out_dim: output tensor channels (dimensions).
        :return: tensorflow.keras.model object
        """
        x = tf.keras.Input(shape=(None, None, in_dim))
        conv = tf.keras.Sequential([tfa.layers.InstanceNormalization(axis=3,
                                                                     center=True,
                                                                     scale=True,
                                                                     beta_initializer="random_uniform",
                                                                     gamma_initializer="random_uniform"),
                                    tf.keras.layers.LeakyReLU(),
                                    tf.keras.layers.Conv2D(in_dim, kernel_size=3, strides=1, padding='same'),
                                    tfa.layers.InstanceNormalization(axis=3,
                                                                     center=True,
                                                                     scale=True,
                                                                     beta_initializer="random_uniform",
                                                                     gamma_initializer="random_uniform"),
                                    tf.keras.layers.LeakyReLU(),
                                    tf.keras.layers.Conv2D(out_dim, kernel_size=3, strides=1, padding='same'),
                                    tf.keras.layers.AveragePooling2D(strides=(2, 2))
                                    ])

        short_cut = tf.keras.Sequential([tf.keras.layers.AveragePooling2D(strides=(2, 2)),
                                         tf.keras.layers.Conv2D(out_dim, kernel_size=1, strides=1)])

        y = conv(x) + short_cut(x)
        return tf.keras.Model(x, y)


class PerceptionRemovalModelComponent:
    @staticmethod
    def get_conv_block(f, s, norm=True, non_linear='leaky_relu'):
        model = keras.Sequential()
        model.add(layers.ZeroPadding2D(padding=(1, 1)))
        model.add(layers.Conv2D(filters=f, kernel_size=(4, 4), strides=(s, s),
                                kernel_initializer=keras.initializers.random_normal(0, 0.02)))

        if norm:
            model.add(layers.BatchNormalization())

        if non_linear == 'leaky_relu':
            model.add(layers.LeakyReLU())

        return model

    @staticmethod
    def get_conv_BN_block(k, d):
        model = keras.Sequential()
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(filters=64, kernel_size=(k, k), dilation_rate=(d, d), padding='same'))
        model.add(layers.LeakyReLU())

        return model
