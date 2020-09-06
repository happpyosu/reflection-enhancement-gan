from Network.component import Component
import tensorflow as tf


class Network:
    """
        This class implements nn models' architecture.
        @Author: chen hao
        @Date:   2020.5.7
    """

    @staticmethod
    def build_generator(img_size=256, noise_dim=4):
        """
        build the generator model
        :param img_size: image size for reflection image R, transmission layer T
        :param noise_dim: noise_dim to concat with the input image (T, R)
        :return: tf.keras.Model object. The generator model accept a 4-D tensor with the shape
        (Batch_size, img_size, img_size, 3 + 3 + noise_dim)
        noted that channel 3 + 3 means the RGB channels for image T and R
        channel noise_dim means the noise channel
        """

        in_layer = tf.keras.layers.Input(shape=(img_size, img_size, 3 + 3 + noise_dim))

        ds1 = Component.get_conv_block(3 + 3 + noise_dim, 32, norm=False)(in_layer)
        ds2 = Component.get_conv_block(32, 64)(ds1)
        ds3 = Component.get_conv_block(64, 128)(ds2)
        ds4 = Component.get_conv_block(128, 256)(ds3)
        ds5 = Component.get_conv_block(256, 256)(ds4)
        ds6 = Component.get_conv_block(256, 256)(ds5)

        us1 = Component.get_deconv_block(256, 256)(ds6)
        us2 = Component.get_deconv_block(512, 256)(tf.concat([us1, ds5], axis=3))
        us3 = Component.get_deconv_block(512, 128)(tf.concat([us2, ds4], axis=3))
        us4 = Component.get_deconv_block(256, 64)(tf.concat([us3, ds3], axis=3))
        us5 = Component.get_deconv_block(128, 32)(tf.concat([us4, ds2], axis=3))
        out_layer = Component.get_deconv_block(64, 3, norm=False, non_linear='tanh')(tf.concat([us5, ds1], axis=3))

        return tf.keras.Model(in_layer, out_layer)

    @staticmethod
    def build_discriminator(img_size=256):
        """
        build the discriminator model.
        :param img_size: image size for synthetic image S
        :return: two tf.keras.Model objects.
        """
        input_layer = tf.keras.layers.Input(shape=(img_size, img_size, 3))

        d1 = tf.keras.Sequential([tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=2),
                                  Component.get_conv_block(3, 32, norm=False),
                                  Component.get_conv_block(32, 64),
                                  Component.get_conv_block(64, 128),
                                  Component.get_conv_block(128, 256, s=1),
                                  Component.get_conv_block(256, 1, s=1, norm=False, non_linear='none')
                                  ])

        d2 = tf.keras.Sequential([Component.get_conv_block(3, 64, norm=False),
                                  Component.get_conv_block(64, 128),
                                  Component.get_conv_block(128, 256),
                                  Component.get_conv_block(256, 1, norm=False, non_linear='none')])

        out1 = d1(input_layer)
        out2 = d2(input_layer)

        return tf.keras.Model(input_layer, (out1, out2))

    @staticmethod
    def build_encoder(img_size=256, noise_dim=4):
        """
        build the encoder model.
        :param img_size: image size for synthetic image S, transmission layer T and reflection layer R.
        :param noise_dim: noise dimension that the encoder will predict.
        :return: tf.keras.Model objects.
        """

        input_layer = tf.keras.layers.Input(shape=(img_size, img_size, 3 + 3 + 3))
        conv = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same')
        res_block = tf.keras.Sequential([Component.get_res_block(64, 128),
                                         Component.get_res_block(128, 192),
                                         Component.get_res_block(192, 256),
                                         Component.get_res_block(256, 256)])
        pool_block = tf.keras.Sequential([tf.keras.layers.LeakyReLU(),
                                          tf.keras.layers.AveragePooling2D(pool_size=(8, 8), padding='same')])

        flatten = tf.keras.layers.Flatten()
        fc_mu = tf.keras.layers.Dense(noise_dim)
        fc_logvar = tf.keras.layers.Dense(noise_dim)

        out = conv(input_layer)
        out = res_block(out)
        out = pool_block(out)
        out = flatten(out)
        mu = fc_mu(out)
        log_var = fc_logvar(out)

        return tf.keras.Model(input_layer, (mu, log_var))

