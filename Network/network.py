from Network.component import Component
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from Network.component import PerceptionRemovalModelComponent, BidirectionalRemovalComponent, MisalignedRemovalComponent, BeyondLinearityComponent


class Network:
    """
        This class implements nn models' architecture for the Reflection-Enhancement-gan
        @Author: chen hao
        @Date:   2020.11.05
    """

    @staticmethod
    def     build_multimodal_D(img_size=256, noise_dim=4):
        """
        build the discriminator model for multi-modal gan.
        :param img_size: image size for synthetic image S and noise
        :return: two tf.keras.Model objects.
        """
        input_layer = tf.keras.layers.Input(shape=(img_size, img_size, 3+noise_dim))

        d1 = tf.keras.Sequential([tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=2),
                                  Component.get_conv_block(3+noise_dim, 32, norm=False),
                                  Component.get_conv_block(32, 64),
                                  Component.get_conv_block(64, 128),
                                  Component.get_conv_block(128, 256, s=1),
                                  Component.get_conv_block(256, 1, s=1, norm=False, non_linear='none')
                                  ])

        d2 = tf.keras.Sequential([Component.get_conv_block(3+noise_dim, 64, norm=False),
                                  Component.get_conv_block(64, 128),
                                  Component.get_conv_block(128, 256),
                                  Component.get_conv_block(256, 1, norm=False, non_linear='none')])

        out1 = d1(input_layer)
        out2 = d2(input_layer)

        return tf.keras.Model(input_layer, (out1, out2))


    @staticmethod
    def build_multimodal_G(img_size=256, noise_dim=4):
        """
        build the generator model
        :param img_size: image size for reflection image R, transmission layer T
        :param noise_dim: noise_dim to concat with the input image (T, R)
        :return: tf.keras.Model object. The generator model accept a 4-D tensor with the shape
        (Batch_size, img_size, img_size, 3 + 3 + noise_dim)
        noted that channel 3 + 3 means the RGB channels for image T and R
        channel noise_dim means the noise channel
        """

        in_layer = tf.keras.layers.Input(shape=(img_size, img_size, 3 + noise_dim))

        ds1 = Component.get_conv_block(3 + noise_dim, 32, norm=False)(in_layer)
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
    def build_optical_synthesis_generator(img_size=256, noise_dim=4):
        """
        build the generator model that use the conventional reflection synthetic model.
        the generator with the optical synthesis prior will only accept a noise-map from the encoder and convert it to
        an (1) alpha blending mask for fusing the transmission layer T and reflection layer R. (2) convolution kernel
        that blurs the reflection layer
        :param img_size: image size for reflection image R, transmission layer T
        :param noise_dim: noise_dim to concat with the input image (T, R)
        :return: tf.keras.Model object. The generator model accepts three 4-D tensors: (1) T. (2) R. (3) noise layer.
        The generator model will output two tensors:
        (1) [alpha_blending_mask] with (256, 256, 3) for mixing two layers.
        (2) [conv-kernel] used for blurring the reflection layer.
        """
        in_layer = tf.keras.layers.Input(shape=(img_size, img_size, 3 + 3 + noise_dim))

        # noise_in = tf.keras.layers.Input(shape=(img_size, img_size, noise_dim))
        # T_in = tf.keras.layers.Input(shape=(img_size, img_size, 3))
        # R_in = tf.keras.layers.Input(shape=(img_size, img_size, 3))
        # split the input tensor
        T_in, R_in, noise_in = tf.split(in_layer, [3, 3, noise_dim], axis=3)
        ds1 = Component.get_conv_block(noise_dim, 32, norm=False)(noise_in)
        ds2 = Component.get_conv_block(32, 64)(ds1)
        ds3 = Component.get_conv_block(64, 128)(ds2)  # d3: (32, 32)
        ds4 = Component.get_conv_block(128, 256)(ds3)
        ds5 = Component.get_conv_block(256, 256)(ds4)
        ds6 = Component.get_conv_block(256, 256)(ds5)

        us1 = Component.get_deconv_block(256, 256)(ds6)
        us2 = Component.get_deconv_block(512, 256)(tf.concat([us1, ds5], axis=3))
        us3 = Component.get_deconv_block(512, 128)(tf.concat([us2, ds4], axis=3))
        us4 = Component.get_deconv_block(256, 64)(tf.concat([us3, ds3], axis=3))  # us4: (64, 64, 64)
        us5 = Component.get_deconv_block(128, 32)(tf.concat([us4, ds2], axis=3))  # us5: (128, 128, 32)

        # let us handle the conv kernel first
        # us5 ---conv--- (32, 32, 16) ---reshape---> (32, 32, 3, 3)
        # (1, 128, 128, 32) -> (1, 64, 64, 16)
        down1 = Component.get_conv_block(32, 16)(us5)

        # (1, 64, 64, 16) -> (1, 32, 32, 9)
        down2 = Component.get_conv_block(16, 9)(down1)

        kernel = tf.reshape(down2, [32, 32, 3, 3])

        # the alpha blending mask
        alpha_mask = Component.get_deconv_block(64, 3, norm=False, non_linear='leaky_relu')(
            tf.concat([us5, ds1], axis=3))
        alpha_mask_sub = layers.subtract([tf.ones_like(alpha_mask), alpha_mask])
        # alpha_mask_sub = Component.get_deconv_block(64, 3, norm=False, non_linear='leaky_relu')(
        #     tf.concat([us5, ds1], axis=3))
        # the blurring kernel
        blurred_R = tf.nn.conv2d(R_in, kernel, strides=[1, 1, 1, 1], padding='SAME')

        # transmission
        t_layer = layers.multiply([T_in, alpha_mask])
        r_layer = layers.multiply([blurred_R, alpha_mask_sub])

        out = layers.add([t_layer, r_layer])

        return tf.keras.Model(in_layer, out)

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
        build the discriminator model for regan.
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
        build the encoder model for regan.
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


class PerceptionRemovalNetworks:
    @staticmethod
    def build_discriminator(img_size=256):
        model = keras.Sequential()
        model.add(layers.InputLayer(input_shape=(None, None, 6)))

        # layer 1
        model.add(PerceptionRemovalModelComponent.get_conv_block(64, 2, non_linear='leaky_relu'))
        # layer 2
        model.add(PerceptionRemovalModelComponent.get_conv_block(128, 2, non_linear='leaky_relu'))
        # layer 3
        model.add(PerceptionRemovalModelComponent.get_conv_block(256, 2, non_linear='leaky_relu'))
        # layer 4
        model.add(PerceptionRemovalModelComponent.get_conv_block(512, 1, non_linear='leaky_relu'))

        # final layer
        model.add(layers.ZeroPadding2D(padding=(1, 1)))
        model.add(layers.Conv2D(filters=1, kernel_size=(4, 4), strides=(1, 1), kernel_initializer=keras.
                                initializers.random_normal(0, 0.02), activation='sigmoid'))

        return model

    @staticmethod
    def build_rm_model():
        inputs = keras.Input(shape=(None, None, 1475), name="image_input")
        model = keras.Sequential()
        model.add(inputs)
        model.add(PerceptionRemovalModelComponent.get_conv_BN_block(1, 1))
        model.add(PerceptionRemovalModelComponent.get_conv_BN_block(3, 1))
        model.add(PerceptionRemovalModelComponent.get_conv_BN_block(3, 2))
        model.add(PerceptionRemovalModelComponent.get_conv_BN_block(3, 4))
        model.add(PerceptionRemovalModelComponent.get_conv_BN_block(3, 8))
        model.add(PerceptionRemovalModelComponent.get_conv_BN_block(3, 16))
        model.add(PerceptionRemovalModelComponent.get_conv_BN_block(3, 32))
        model.add(PerceptionRemovalModelComponent.get_conv_BN_block(3, 64))
        model.add(PerceptionRemovalModelComponent.get_conv_BN_block(3, 1))
        model.add(layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same'))

        return model


class BidirectionalRemovalNetworks:
    @staticmethod
    def build_vanilla_generator():
        inputs = keras.Input(shape=(256, 256, 3))

        # the input layer
        x = layers.Conv2D(filters=64, kernel_size=(4, 4), padding='same')(inputs)
        x = layers.LeakyReLU()(x)

        c1 = BidirectionalRemovalComponent.get_conv_block(512, 2)(x)
        c2 = BidirectionalRemovalComponent.get_conv_block(256, 2)(c1)
        c3 = BidirectionalRemovalComponent.get_conv_block(128, 2)(c2)
        c4 = BidirectionalRemovalComponent.get_conv_block(64, 2)(c3)
        c5 = BidirectionalRemovalComponent.get_conv_block(32, 2)(c4)
        c6 = BidirectionalRemovalComponent.get_conv_block(16, 2)(c5)
        c7 = BidirectionalRemovalComponent.get_conv_block(8, 2)(c6)

        d1 = BidirectionalRemovalComponent.get_deconv_block(8, 2)(c7)
        d2 = BidirectionalRemovalComponent.get_deconv_block(16, 2)(tf.concat([d1, c6], axis=3))
        d3 = BidirectionalRemovalComponent.get_deconv_block(32, 2)(tf.concat([d2, c5], axis=3))
        d4 = BidirectionalRemovalComponent.get_deconv_block(64, 2)(tf.concat([d3, c4], axis=3))
        d5 = BidirectionalRemovalComponent.get_deconv_block(128, 2)(tf.concat([d4, c3], axis=3))
        d6 = BidirectionalRemovalComponent.get_deconv_block(256, 2)(tf.concat([d5, c2], axis=3))
        d7 = BidirectionalRemovalComponent.get_deconv_block(512, 2)(tf.concat([d6, c1], axis=3))

        # the output layer
        y = layers.Conv2DTranspose(filters=3, kernel_size=(4, 4), padding='same', activation=keras.activations.tanh)(d7)

        return keras.Model(inputs, y)

    @staticmethod
    def build_bidirectional_unit():
        inputs = keras.Input(shape=(256, 256, 6))

        # the input layer
        x = layers.Conv2D(filters=64, kernel_size=(4, 4), padding='same')(inputs)
        x = layers.LeakyReLU()(x)

        c1 = BidirectionalRemovalComponent.get_conv_block(512, 2)(x)
        c2 = BidirectionalRemovalComponent.get_conv_block(256, 2)(c1)
        c3 = BidirectionalRemovalComponent.get_conv_block(128, 2)(c2)
        c4 = BidirectionalRemovalComponent.get_conv_block(64, 2)(c3)
        c5 = BidirectionalRemovalComponent.get_conv_block(32, 2)(c4)

        d1 = BidirectionalRemovalComponent.get_deconv_block(32, 2)(c5)
        d2 = BidirectionalRemovalComponent.get_deconv_block(64, 2)(tf.concat([d1, c4], axis=3))
        d3 = BidirectionalRemovalComponent.get_deconv_block(128, 2)(tf.concat([d2, c3], axis=3))
        d4 = BidirectionalRemovalComponent.get_deconv_block(256, 2)(tf.concat([d3, c2], axis=3))
        d5 = BidirectionalRemovalComponent.get_deconv_block(512, 2)(tf.concat([d4, c1], axis=3))

        # the output layer
        y = layers.Conv2DTranspose(filters=3, kernel_size=(4, 4), padding='same', activation=keras.activations.tanh)(d5)

        return keras.Model(inputs, y)

    @staticmethod
    def build_discriminator(img_size=256):
        model = keras.Sequential()
        model.add(layers.InputLayer(input_shape=(None, None, 3)))

        # layer 1
        model.add(PerceptionRemovalModelComponent.get_conv_block(64, 2, non_linear='leaky_relu'))
        # layer 2
        model.add(PerceptionRemovalModelComponent.get_conv_block(128, 2, non_linear='leaky_relu'))
        # layer 3
        model.add(PerceptionRemovalModelComponent.get_conv_block(256, 2, non_linear='leaky_relu'))
        # layer 4
        model.add(PerceptionRemovalModelComponent.get_conv_block(512, 1, non_linear='leaky_relu'))

        # final layer
        model.add(layers.ZeroPadding2D(padding=(1, 1)))
        model.add(layers.Conv2D(filters=1, kernel_size=(4, 4), strides=(1, 1), kernel_initializer=keras.
                                initializers.random_normal(0, 0.02), activation='sigmoid'))

        return model


class MisalignedRemovalNetworks:
    @staticmethod
    def build_DRNet(in_dims):
        """
        build the reflection removal network
        :param in_dims: input feature map dimensions. origin paper uses the VGG19 to extract high-level image features.
        :return: tf.keras.Model.
        """
        n_res = 13

        model = keras.Sequential()

        # conv1 256 -> 256
        model.add(MisalignedRemovalComponent.get_conv_block(in_dim=in_dims, f=64, k=3, s=1, d=1,
                                                            norm=True, non_linear='relu'))
        # conv2 256 -> 256
        model.add(MisalignedRemovalComponent.get_conv_block(in_dim=64, f=128, k=3, s=1, d=1,
                                                            norm=True, non_linear='relu'))
        # conv3 256 -> 128
        model.add(MisalignedRemovalComponent.get_conv_block(in_dim=128, f=256, k=3, s=2, d=1,
                                                            norm=True, non_linear='relu'))

        # 13 res blocks 128 -> 128
        for _ in range(n_res):
            model.add(MisalignedRemovalComponent.get_res_block(f=256, d=1, sz=128))

        # deconv1 128 -> 256
        model.add(MisalignedRemovalComponent.get_deconv_block(in_dim=256, f=256, k=3, s=2, d=1))

        # deconv2 256 -> 256
        model.add(MisalignedRemovalComponent.get_deconv_block(in_dim=256, f=256, k=3, s=1, d=1))

        # pyramid pooling 256 -> 256
        model.add(MisalignedRemovalComponent.get_pyramid_pooling_block(feat_sz=256, in_channels=256,
                                                                       out_channels=256, ct_channels=64))

        # deconv3 256 -> 256
        model.add(MisalignedRemovalComponent.get_deconv_block(in_dim=256, f=256, k=3, s=1, d=1))

        return model

    @staticmethod
    def build_patch_gan_discriminator():
        inp = keras.Input(shape=(256, 256, 3))
        model = keras.Sequential()
        n_layers = 3

        model.add(inp)

        model.add(layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())

        nf = 64
        for n in range(n_layers):
            model.add(layers.Conv2D(filters=nf, kernel_size=4, strides=(2, 2), padding='same'))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())
            nf = min(nf * 2, 512)

        # 256 / 8 = 32

        # 32 -> 32
        model.add(layers.Conv2D(filters=nf, kernel_size=4, strides=(1, 1), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        # 32 -> 32
        model.add(layers.Conv2D(filters=nf, kernel_size=4, strides=(1, 1), padding='same', activation='sigmoid'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        return model


class BeyondLinearityNetworks:
    @staticmethod
    def build_SynNet(img_size=256):
        inp = layers.Input(shape=(img_size, img_size, 6))

        t_layer, r_layer = tf.split(inp, num_or_size_splits=2)

        # 256 -> 128
        conv1 = BeyondLinearityComponent.get_conv_block(6, 64)(inp)

        # 128 -> 64
        conv2 = BeyondLinearityComponent.get_conv_block(64, 128)(conv1)

        # 64 -> 32
        conv3 = BeyondLinearityComponent.get_conv_block(128, 256)(conv2)

        x = conv3
        # 32 -> 32
        for _ in range(9):
            x = BeyondLinearityComponent.get_res_block(256, 256)(x)

        # 32 -> 64
        deconv1 = BeyondLinearityComponent.get_deconv_block(256, 128)(x)

        # 64 -> 128
        deconv2 = BeyondLinearityComponent.get_deconv_block(128, 64)(deconv1)

        # 128 -> 256 and get the alpha blending mask.
        mask = BeyondLinearityComponent.get_deconv_block(64, 3, non_linear='sigmoid')(deconv2)
        mask_d = tf.ones_like(mask) - mask

        out_t = layers.multiply()([mask, t_layer])
        out_r = layers.multiply()([mask_d, r_layer])

        out = out_t + out_r

        return keras.Model(inp, [mask, out])

    @staticmethod
    def build_RmNet(img_size=256):
        inp = layers.Input(shape=(img_size, img_size, 3))

        # 256 -> 128
        conv1 = BeyondLinearityComponent.get_conv_block(3, 16)(inp)

        # 128 -> 64
        conv2 = BeyondLinearityComponent.get_conv_block(16, 32)(conv1)

        # 64 -> 32
        conv3 = BeyondLinearityComponent.get_conv_block(32, 64)(conv2)

        # 32 -> 16
        conv4 = BeyondLinearityComponent.get_conv_block(64, 128)(conv3)

        # 16 -> 8
        conv5 = BeyondLinearityComponent.get_conv_block(128, 256)(conv4)

        # 8 -> 4
        conv6 = BeyondLinearityComponent.get_conv_block(256, 512)(conv5)

        def get_upsampling_unit(non_linear):
            # 4 -> 8
            deconv1 = BeyondLinearityComponent.get_deconv_block(512, 256)(conv6)

            # 8 -> 16
            deconv2 = BeyondLinearityComponent.get_deconv_block(256, 128)(tf.concat([deconv1, conv5], axis=3))

            # 16 -> 32
            deconv3 = BeyondLinearityComponent.get_deconv_block(128, 64)(tf.concat([deconv2, conv4], axis=3))

            # 32 -> 64
            deconv4 = BeyondLinearityComponent.get_deconv_block(64, 32)(tf.concat([deconv3, conv3], axis=3))

            # 64 -> 128
            deconv5 = BeyondLinearityComponent.get_deconv_block(32, 16)(tf.concat([deconv4, conv2], axis=3))

            # 128 -> 256
            out = BeyondLinearityComponent.get_deconv_block(16, 3, non_linear=non_linear)(tf.concat([deconv5, conv1], axis=3))

            return out

        t_layer = get_upsampling_unit('tanh')
        r_layer = get_upsampling_unit('tanh')
        mask = get_upsampling_unit('sigmoid')
        mask_d = tf.ones_like(mask) - mask

        recombined_image = layers.multiply([mask, t_layer]) + layers.multiply([mask_d, r_layer])

        return keras.Model(inp, [t_layer, r_layer, recombined_image])

    @staticmethod
    def build_discriminator():
        """
        No implement clue for the discriminator, so we simply use a patch-gan discriminator.
        :param self:
        :return:
        """
        return MisalignedRemovalNetworks.build_patch_gan_discriminator()


class Vgg19FeatureExtractor:
    @staticmethod
    def build_vgg19_feature_extractor():
        """
        build the VGG19 submodel used for building perceptual middle features.
        :return: tf.keras.Model
        """
        vgg19 = keras.applications.VGG19()
        features = [layer.output for layer in vgg19.layers]
        conv1_2 = features[2]
        conv2_2 = features[5]
        conv3_2 = features[8]
        conv4_2 = features[13]
        conv5_2 = features[18]
        features_list = [conv1_2, conv2_2, conv3_2, conv4_2, conv5_2]

        return keras.Model(vgg19.input, features_list)



if __name__ == '__main__':
    extractor = Vgg19FeatureExtractor.build_vgg19_feature_extractor()
    extractor.summary()