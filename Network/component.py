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


class BidirectionalRemovalComponent:
    @staticmethod
    def get_conv_block(f, s, norm=True, non_linear='leaky_relu'):
        model = keras.Sequential()
        model.add(layers.Conv2D(filters=f, kernel_size=(4, 4), strides=(s, s), padding='same'))
        if norm:
            model.add(layers.BatchNormalization())

        if non_linear == 'leaky_relu':
            model.add(layers.LeakyReLU())

        return model

    @staticmethod
    def get_deconv_block(f, s, norm=True, non_linear='relu'):
        model = keras.Sequential()
        model.add(layers.Conv2DTranspose(f, kernel_size=(4, 4), strides=(s, s), padding='same'))
        if norm:
            model.add(layers.BatchNormalization())

        if non_linear == 'relu':
            model.add(layers.ReLU())

        return model


class MisalignedRemovalComponent:
    @staticmethod
    def get_SE_block(in_dim, sz, c, reduction=16):
        """
        get the squeeze-and-Excitation Network block
        :param in_dim: channels of the input feature map.
        :param sz: the size of input feature map, assuming the feature map is squared.
        :param c: channel numbers
        :param reduction: reduction rate on fc layer
        :return: tf.keras.Model Object
        """
        inp = layers.Input(shape=(None, None, in_dim))

        # pooling to one-dim per channel
        p1 = layers.AveragePooling2D(pool_size=(sz, sz))(inp)

        # flatten to one-dimension vector for fc.
        fl = layers.Flatten()(p1)

        # fc1: c -> c // reduction
        fc1 = layers.Dense(c // reduction, activation=keras.activations.relu)(fl)

        # fc2: c // reduction -> c
        fc2 = layers.Dense(c, activation=keras.activations.sigmoid)(fc1)

        # reshape to (b, 1, 1, c)
        re = layers.Reshape((1, 1, c))(fc2)

        # re-weight the feature map
        out = inp * re

        return keras.Model(inp, out)

    @staticmethod
    def get_pyramid_pooling_block(feat_sz, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=1):
        """
        get the pyramid pooling block
        :param in_channels: channels of the input feature map.
        :param feat_sz: the feature size
        :param out_channels: feature channel nums
        :param scales: scales list
        :param ct_channels: target channel
        :return: tf.keras.Model
        """

        def make_stage(scale, c):
            model = keras.Sequential()
            model.add(layers.AveragePooling2D(pool_size=(scale, scale)))
            model.add(layers.Conv2D(filters=c, kernel_size=1, use_bias=False))
            model.add(layers.LeakyReLU())
            return model

        inp = keras.Input(shape=(feat_sz, feat_sz, in_channels))

        # pooling list for different size pooling layer
        pool_list = [make_stage(sz, ct_channels)(inp) for sz in scales]
        bottleneck = layers.Conv2D(filters=out_channels, kernel_size=1)

        priors = tf.concat([tf.image.resize(images=stage,
                                            size=(feat_sz, feat_sz),
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) for stage in pool_list], axis=3)
        out = bottleneck(priors)

        return keras.Model(inp, out)

    @staticmethod
    def get_conv_block(in_dim, f, k, s, d, norm=True, non_linear='leaky_relu'):
        """
        get a convolution blocks
        :param in_dim: input feature map dimensions
        :param f: filter nums
        :param k: kernel size
        :param s: stride
        :param d: dilation rate
        :param norm: use normalization or not
        :param non_linear: non-linear activation function
        :return: tf.keras.Model
        """
        model = keras.Sequential()
        model.add(layers.Input(shape=(None, None, in_dim)))
        model.add(layers.Conv2D(f, k, (s, s), dilation_rate=(d, d), padding='same'))
        if norm:
            model.add(layers.BatchNormalization())
        if non_linear == 'leaky_relu':
            model.add(layers.LeakyReLU())
        elif non_linear == 'relu':
            model.add(layers.ReLU())
        else:
            pass

        return model

    @staticmethod
    def get_deconv_block(in_dim, f, k, s, d, norm=True, non_linear='leaky_relu'):
        """
        get a de-convolution blocks
        :param in_dim: input feature map dimensions
        :param f: filter nums
        :param k: kernel size
        :param s: stride
        :param d: dilation rate
        :param norm: use normalization or not
        :param non_linear: non-linear activation function
        :return: tf.keras.Model
        """
        model = keras.Sequential()
        model.add(layers.Input(shape=(None, None, in_dim)))
        model.add(layers.Conv2DTranspose(f, k, (s, s), dilation_rate=(d, d), padding='same'))
        if norm:
            model.add(layers.BatchNormalization())
        if non_linear == 'leaky_relu':
            model.add(layers.LeakyReLU())
        elif non_linear == 'relu':
            model.add(layers.ReLU())
        else:
            pass

        return model

    @staticmethod
    def get_res_block(f, d, sz, norm=True, non_linear='relu', se_reduction=True, res_scale=1):
        """
        get a residual linked nn block.
        :param f: channels of the input filters and output filters
        :param d: dilation rate
        :param sz: input feature map size
        :param norm: use normalization or not
        :param non_linear: non_linear function type.
        :param se_reduction: use SE-reduction or not
        :param res_scale: res scale rate
        :return: tf.keras.Model.
        """
        conv1 = MisalignedRemovalComponent.get_conv_block(in_dim=f, f=f, k=3, s=1, d=d, norm=norm, non_linear=non_linear)
        conv2 = MisalignedRemovalComponent.get_conv_block(in_dim=f, f=f, k=3, s=1, d=d, norm=norm, non_linear='none')

        # in-dim = out-dim = f
        inp = keras.Input(shape=(None, None, f))

        res = inp
        out = conv1(inp)
        out = conv2(out)

        if se_reduction:
            se = MisalignedRemovalComponent.get_SE_block(in_dim=f, sz=sz, c=f)
            out = se(out)

        out = out * res_scale
        out = out + res

        return keras.Model(inp, out)



if __name__ == '__main__':
    b = MisalignedRemovalComponent.get_pyramid_pooling_block(feat_sz=256, in_channels=256, out_channels=256, ct_channels=256)
    b.summary()


