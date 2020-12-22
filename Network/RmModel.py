import tensorflow as tf
from Network.network import Network, PerceptionRemovalNetworks, BidirectionalRemovalNetworks, Vgg19FeatureExtractor, \
    MisalignedRemovalNetworks, EncoderDecoderRemovalNetworks, BeyondLinearityNetworks
from Dataset.dataset import DatasetFactory
from utils.imageUtils import ImageUtils
'''
This file offers some reflection removal model implements.
(1) PerceptionRemovalModel: reflection removal with the perception loss
(2) BidirectionalRemovalModel: reflection removal with the bidirectional translation
(3) MisalignedRemovalModel: reflection removal with misaligned data and network enhancement(channel attention).
(4) EncoderDecoderRemovalModel: Single Image Reflection Removal Using Deep Encoder-Decoder Network

and some jointly generative and removal models. 
(5) BeyondLinearityRemovalModel: Single Image Reflection Removal Beyond Linearity using a predicted alpha blending mask
to syn the training data.
@author: chen hao
@date: 2020-11-11
'''

from tensorflow import keras


class PerceptionRemovalModel:
    """
    reflection removal with the perception loss.
    detail info about this paper please refer to:
    Zhang, Xuaner, Ren Ng, and Qifeng Chen. "Single image reflection separation with perceptual losses.
    " Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
    """

    def __init__(self):
        # epoch
        self.EPOCH = 100

        # epsilon for log function
        self.EPS = 1e-12

        # the image size
        self.img_size = 256

        # the vgg19 feature extractor
        self.feature_extractor = Vgg19FeatureExtractor.build_vgg19_feature_extractor()

        # the rm model
        self.rm = PerceptionRemovalNetworks.build_rm_model()

        # the d model
        self.d = PerceptionRemovalNetworks.build_discriminator()

        # optimizer for g and d
        self.g_optimizer = keras.optimizers.Adam(learning_rate=0.002)
        self.d_optimizer = keras.optimizers.Adam(learning_rate=0.001)

        # training dataset and test dataset
        self.train_dataset = DatasetFactory.get_dataset_by_name(name="RealDataset", mode="train", batch_size=4)
        self.val_dataset = DatasetFactory.get_dataset_by_name(name="RealDataset", mode='val')

        # config logging
        self.inc = 0
        self.save_every = 10
        self.output_every = 2

    def start_train_task(self):
        for _ in range(self.EPOCH):
            self.inc += 1
            for (t, r, m) in self.train_dataset:
                self.train_one_step(t, r, m)

            if self.inc % self.output_every:
                self.output_middle_result()

            if self.inc & self.save_every:
                self.save_weights()

    def forward(self, m):
        # extract hyper-col feature
        features_list = self.feature_extractor(m)
        features = m

        for f in features_list:
            resized = tf.image.resize(f, (self.img_size, self.img_size))
            features = tf.concat([features, resized], axis=3)

        # forward
        pred = self.rm(features, training=True)
        pred_t, pred_r = tf.split(pred, num_or_size_splits=2, axis=3)
        return pred_t

    def save_weights(self):
        self.rm.save_weights('../save/' + 'percpRm_' + str(self.inc) + '.h5')

    def load_weights(self, epoch: int):
        self.rm.load_weights('../save/' + 'percpRm_' + str(epoch) + '.h5')

    def output_middle_result(self, rows=5):
        iter = self.val_dataset.__iter__()
        img_lists = []
        for _ in range(rows):
            img_list = []
            t, r, m = next(iter)
            t1 = tf.squeeze(t, axis=0)
            r1 = tf.squeeze(r, axis=0)
            m1 = tf.squeeze(m, axis=0)
            img_list.append(t1)
            img_list.append(r1)
            img_list.append(m1)
            # extract hyper-col feature
            features_list = self.feature_extractor(m)
            features = m

            for f in features_list:
                resized = tf.image.resize(f, (self.img_size, self.img_size))
                features = tf.concat([features, resized], axis=3)

            # forward
            pred = self.rm(features, training=True)
            pred_t, pred_r = tf.split(pred, num_or_size_splits=2, axis=3)
            pred_t = tf.squeeze(pred_t, axis=0)
            pred_r = tf.squeeze(pred_r, axis=0)

            del features_list

            img_list.append(pred_t)
            img_list.append(pred_r)
            img_lists.append(img_list)

        ImageUtils.plot_images(rows, 3 + 2, img_lists, is_save=True, epoch_index=self.inc)

    @tf.function
    def train_one_step(self, t, r, m):
        # obtain the hypercolumn features first.
        features_list = self.feature_extractor(m)
        features = m

        for f in features_list:
            resized = tf.image.resize(f, (self.img_size, self.img_size))
            features = tf.concat([features, resized], axis=3)

        # deallocate the big tensors
        del features_list

        with tf.GradientTape(watch_accessed_variables=False) as g_tape, \
                tf.GradientTape(watch_accessed_variables=False) as d_tape:
            g_tape.watch(self.rm.trainable_variables)
            d_tape.watch(self.d.trainable_variables)

            # forward
            pred = self.rm(features, training=True)
            pred_t, pred_r = tf.split(pred, num_or_size_splits=2, axis=3)

            # perceptual losses
            perceptual_loss = self.compute_perceptual_loss(pred_t, t) + self.compute_perceptual_loss(pred_r, r)

            # gan loss
            real = tf.concat([m, t], axis=3)
            fake = tf.concat([m, pred_t], axis=3)

            on_real = self.d(real)
            on_fake = self.d(fake)

            loss_d = 0.5 * tf.reduce_mean(-tf.math.log(on_real + self.EPS) + tf.math.log(1 - on_fake + self.EPS))
            loss_g = tf.reduce_mean(tf.math.log(1 - on_fake + self.EPS))

            # exclusion loss
            loss_grad_x, loss_grad_y = self.compute_exclusion_loss(pred_t, pred_r)
            loss_grad = tf.reduce_sum(sum(loss_grad_x) / 3) + tf.reduce_sum(sum(loss_grad_y) / 3)

            # total toss
            loss = 0.2 * perceptual_loss + loss_grad

            # update g
            Loss_rm = loss * 100 + loss_g
            grad_rm = g_tape.gradient(Loss_rm, self.rm.trainable_variables)
            self.g_optimizer.apply_gradients(zip(grad_rm, self.rm.trainable_variables))

            # update d
            grad_d = d_tape.gradient(loss_d, self.d.trainable_variables)
            self.d_optimizer.apply_gradients(zip(grad_d, self.d.trainable_variables))

    def compute_img_gradient(self, img):
        """
        compute the image's gradient on both x and y directions.
        :param img:
        :return: tf.Tensor
        """
        grad_x = img[:, 1:, :, :] - img[:, :-1, :, :]
        grad_y = img[:, :, 1:, :] - img[:, :, :-1, :]
        return grad_x, grad_y

    def compute_l1_loss(self, img1, img2):
        return tf.reduce_mean(tf.abs(img1 - img2))

    def compute_perceptual_loss(self, img1, img2):
        f1 = self.feature_extractor(img1)
        f2 = self.feature_extractor(img2)

        # l1 loss
        loss = self.compute_l1_loss(img1, img2)

        # perceptual loss
        for fe1, fe2 in zip(f1, f2):
            loss += self.compute_l1_loss(fe1, fe2)

        return loss

    def compute_exclusion_loss(self, img1, img2):
        """
        compute the exclusion loss of the input image img1 and img2
        :param img1: image 1
        :param img2: image 2
        :return: tf.Tensor
        """
        gradx_loss = []
        grady_loss = []

        gradx1, grady1 = self.compute_img_gradient(img1)
        gradx2, grady2 = self.compute_img_gradient(img2)
        alphax = 2.0 * tf.reduce_mean(tf.abs(gradx1)) / tf.reduce_mean(tf.abs(gradx2))
        alphay = 2.0 * tf.reduce_mean(tf.abs(grady1)) / tf.reduce_mean(tf.abs(grady2))

        gradx1_s = (tf.nn.sigmoid(gradx1) * 2) - 1
        grady1_s = (tf.nn.sigmoid(grady1) * 2) - 1
        gradx2_s = (tf.nn.sigmoid(gradx2 * alphax) * 2) - 1
        grady2_s = (tf.nn.sigmoid(grady2 * alphay) * 2) - 1

        gradx_loss.append(
            tf.reduce_mean(tf.multiply(tf.square(gradx1_s), tf.square(gradx2_s)), axis=[1, 2, 3]) ** 0.25)
        grady_loss.append(
            tf.reduce_mean(tf.multiply(tf.square(grady1_s), tf.square(grady2_s)), axis=[1, 2, 3]) ** 0.25)

        return gradx_loss, grady_loss


class BidirectionalRemovalModel:
    """
    Seeing Deeply and Bidirectionally: A Deep Learning
    Approach for Single Image Reflection Removal
    """

    def __init__(self):
        # training EPOCH
        self.EPOCH = 100

        # epsilon for log function
        self.EPS = 1e-12

        # image size
        self.img_size = 256

        # generator (g0)
        self.g0 = BidirectionalRemovalNetworks.build_vanilla_generator()

        # two bidirectional unit (H and g1)
        self.H = BidirectionalRemovalNetworks.build_bidirectional_unit()
        self.g1 = BidirectionalRemovalNetworks.build_bidirectional_unit()

        # the discriminator
        self.d = BidirectionalRemovalNetworks.build_discriminator()

        # optimizer for g and d
        self.g0_optimizer = keras.optimizers.Adam(learning_rate=0.0002)
        self.g1_optimizer = keras.optimizers.Adam(learning_rate=0.0002)
        self.d_optimizer = keras.optimizers.Adam(learning_rate=0.0002)
        self.H_optimizer = keras.optimizers.Adam(learning_rate=0.0002)

        # training dataset and test dataset
        self.train_dataset = DatasetFactory.get_dataset_by_name(name="RealDataset", mode="train", batch_size=20)
        self.val_dataset = DatasetFactory.get_dataset_by_name(name="RealDataset", mode='val')

        # config logging
        self.inc = 0
        self.save_every = 10
        self.output_every = 2

    def save_weights(self):
        self.g0.save_weights('../save/' + 'bidirRm_g0_' + str(self.inc) + '.h5')
        self.H.save_weights('../save/' + 'bidirRm_H_' + str(self.inc) + '.h5')
        self.g1.save_weights('../save/' + 'bidirRm_g1_' + str(self.inc) + '.h5')

    def load_weights(self, epoch: int):
        self.g0.load_weights('../save/' + 'bidirRm_g0_' + str(epoch) + '.h5')
        self.H.load_weights('../save/' + 'bidirRm_H_' + str(epoch) + '.h5')
        self.g1.load_weights('../save/' + 'bidirRm_g1_' + str(epoch) + '.h5')

    def forward(self, m):
        pred_B = self.g0(m)
        IB = tf.concat([m, pred_B], axis=3)
        pred_R = self.H(IB)
        IR = tf.concat([m, pred_R], axis=3)
        pred_B1 = self.g1(IR)

        return pred_B1

    def start_train_task(self):
        for _ in range(self.EPOCH):
            self.inc += 1
            for t, r, m in self.train_dataset:
                self.train_one_step(t, r, m)

            if self.inc % self.save_every:
                self.save_weights()

            if self.inc % self.output_every:
                self.output_middle_result()

    def output_middle_result(self, rows=5):
        iter = self.val_dataset.__iter__()
        img_lists = []
        for _ in range(rows):
            img_list = []
            t, r, m = next(iter)
            t1 = tf.squeeze(t, axis=0)
            r1 = tf.squeeze(r, axis=0)
            m1 = tf.squeeze(m, axis=0)
            img_list.append(t1)
            img_list.append(r1)
            img_list.append(m1)

            pred_B = self.g0(m)
            IB = tf.concat([m, pred_B], axis=3)
            pred_R = self.H(IB)
            IR = tf.concat([m, pred_R], axis=3)
            pred_B1 = self.g1(IR)

            B0 = tf.squeeze(pred_B, axis=0)
            R = tf.squeeze(pred_R, axis=0)
            B1 = tf.squeeze(pred_B1, axis=0)

            img_list.append(B0)
            img_list.append(R)
            img_list.append(B1)

            img_lists.append(img_list)

        ImageUtils.plot_images(rows, 3 + 3, img_lists, is_save=True, epoch_index=self.inc)

    @tf.function
    def train_one_step(self, t, r, m):
        with tf.GradientTape() as g0_tape, tf.GradientTape() as d_tape, \
                tf.GradientTape() as g1_tape, tf.GradientTape() as H_tape:

            # pass m to g0
            pred_B = self.g0(m, training=True)

            # concat I with B, pass IB to H
            IB = tf.concat([m, pred_B], axis=3)
            pred_R = self.H(IB, training=True)

            # concat I with R, pass IR to g1
            IR = tf.concat([m, pred_R], axis=3)
            pred_B1 = self.g1(IR, training=True)

            # l2 loss for g0
            loss_g0 = tf.reduce_mean((pred_B - t) ** 2)

            # l2 loss for H
            loss_H = tf.reduce_mean((pred_R - r) ** 2)

            # l2 loss for g1
            loss_g1 = tf.reduce_mean((pred_B1 - t) ** 2)

            # training d
            on_fake = self.d(pred_B1)
            on_real = self.d(t)
            Loss_d = tf.reduce_mean(tf.math.log(on_real + self.EPS) + tf.math.log(1 - on_fake + self.EPS))
            grad_d = d_tape.gradient(Loss_d, self.d.trainable_variables)
            self.d_optimizer.apply_gradients(zip(grad_d, self.d.trainable_variables))

            # gan loss for g1
            on_fake = self.d(pred_B1)
            loss_gan = tf.reduce_mean(-tf.math.log(on_real))

            Loss_all = loss_g0 + loss_g1 + loss_H + loss_gan

            # update g1
            grad_g1 = g1_tape.gradient(Loss_all, self.g1.trainable_variables)
            self.g1_optimizer.apply_gradients(zip(grad_g1, self.g1.trainable_variables))

            # update H
            grad_H = H_tape.gradient(Loss_all, self.H.trainable_variables)
            self.H_optimizer.apply_gradients(zip(grad_H, self.H.trainable_variables))

            # update g0
            grad_g0 = g0_tape.gradient(Loss_all, self.g0.trainable_variables)
            self.g0_optimizer.apply_gradients(zip(grad_g0, self.g0.trainable_variables))


class MisalignedRemovalModel:
    def __init__(self):
        # epsilon for log function
        self.EPS = 1e-12

        # the image size
        self.img_size = 256

        # the vgg19 feature extractor
        self.feature_extractor = Vgg19FeatureExtractor.build_vgg19_feature_extractor()

        # vgg features 1472 + origin image 3 = 1475
        self.rm = MisalignedRemovalNetworks.build_DRNet(in_dims=1475)

        # training dataset and test dataset
        self.train_dataset = DatasetFactory.get_dataset_by_name(name="RealDataset", mode="train", batch_size=4)
        self.val_dataset = DatasetFactory.get_dataset_by_name(name="RealDataset", mode='val')

        # d
        self.d = MisalignedRemovalNetworks.build_patch_gan_discriminator()

        # optimizer for g and d
        self.g_optimizer = keras.optimizers.Adam(learning_rate=0.0002)
        self.d_optimizer = keras.optimizers.Adam(learning_rate=0.0002)

        # config
        self.EPOCH = 100
        self.inc = 0
        self.save_every = 10
        self.output_every = 2

    def start_train_task(self):
        for _ in range(self.EPOCH):
            self.inc += 1

            for t, r, m in self.train_dataset:
                self.train_one_step(t, r, m)

            if self.inc % self.save_every == 0:
                    self.save_weights()

            if self.inc % self.output_every == 0:
                    self.output_middle_result()

    def save_weights(self):
        self.rm.save_weights('../save/' + 'misalignedRm_' + str(self.inc) + '.h5')

    def output_middle_result(self, rows=5):
        iter = self.val_dataset.__iter__()
        img_lists = []
        for _ in range(rows):
            img_list = []
            t, r, m = next(iter)
            t1 = tf.squeeze(t, axis=0)
            r1 = tf.squeeze(r, axis=0)
            m1 = tf.squeeze(m, axis=0)
            img_list.append(t1)
            img_list.append(r1)
            img_list.append(m1)
            # extract hyper-col feature
            features_list = self.feature_extractor(m)
            features = m

            for f in features_list:
                resized = tf.image.resize(f, (self.img_size, self.img_size))
                features = tf.concat([features, resized], axis=3)

            # forward
            pred_t = self.rm(features, training=True)
            pred_t = tf.squeeze(pred_t, axis=0)

            del features_list
            img_list.append(pred_t)
            img_lists.append(img_list)

        ImageUtils.plot_images(rows, 3 + 1, img_lists, is_save=True, epoch_index=self.inc)

    @tf.function
    def train_one_step(self, t, r, m):
        # obtain the hypercolumn features first.
        features_list = self.feature_extractor(m)
        features = m

        for f in features_list:
            resized = tf.image.resize(f, (self.img_size, self.img_size))
            features = tf.concat([features, resized], axis=3)

        # deallocate the big tensors
        del features_list
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            # train rm first
            pred_t = self.rm(features, training=True)

            # the feature loss
            loss_feature = self.compute_feature_loss(pred_t, t)

            # the pixel loss
            loss_pixel = self.compute_pixel_loss(pred_t, t)

            # the gan loss
            loss_gan = tf.reduce_mean(-tf.math.log(self.d(pred_t) + self.EPS))

            # total loss
            Loss_total = 0.1 * loss_feature + loss_pixel + 0.01 * loss_gan

            grad_g = g_tape.gradient(Loss_total, self.rm.trainable_variables)
            self.g_optimizer.apply_gradients(zip(grad_g, self.rm.trainable_variables))

            # train d
            pred_t = self.rm(features, training=True)
            on_real = self.d(t)
            on_fake = self.d(pred_t)

            Loss_d = tf.reduce_mean(tf.math.log(on_real + self.EPS) + tf.math.log(1 - on_fake + self.EPS))

            grad_d = d_tape.gradient(Loss_d, self.d.trainable_variables)
            self.d_optimizer.apply_gradients(zip(grad_d, self.d.trainable_variables))

    def compute_feature_loss(self, img1, img2):
        """
        compute the feature using vgg19
        :param img1: image1
        :param img2: image2
        :return: loss tensor
        """
        feat_list_img1 = self.feature_extractor(img1)
        feat_list_img2 = self.feature_extractor(img2)
        feature_loss = 0
        for i in range(len(feat_list_img1)):
            feature_loss += tf.reduce_mean(tf.abs(feat_list_img1[i] - feat_list_img2[i]))

        return feature_loss

    def compute_pixel_loss(self, img1, img2):
        """
        compute the pixel loss (including the l1 loss and gradient loss on two images)
        :param img2: image2
        :param img1: image1
        :return: tf.Tensor
        """
        l1_loss = tf.reduce_mean(tf.abs(img1 - img2))

        grad_x_img1 = img1[:, 1:, :, :] - img1[:, :-1, :, :]
        grad_y_img1 = img1[:, :, 1:, :] - img1[:, :, :-1, :]

        grad_x_img2 = img2[:, 1:, :, :] - img2[:, :-1, :, :]
        grad_y_img2 = img2[:, :, 1:, :] - img2[:, :, :-1, :]

        grad_loss = tf.reduce_mean(tf.abs(grad_x_img1 - grad_x_img2)) + tf.reduce_mean(tf.abs(grad_y_img1 - grad_y_img2))

        return l1_loss + grad_loss


class EncoderDecoderRemovalModel:
    def __init__(self):
        # the image size
        self.img_size = 256

        # the vgg19 feature extractor
        self.feature_extractor = Vgg19FeatureExtractor.build_vgg19_feature_extractor()

        # training dataset and test dataset
        self.train_dataset = DatasetFactory.get_dataset_by_name(name="RealDataset", mode="train", batch_size=4)
        self.val_dataset = DatasetFactory.get_dataset_by_name(name="RealDataset", mode='val')

        # optimizer
        self.g_optimizer = keras.optimizers.Adam(learning_rate=0.0002)

        # rm model
        self.rm = EncoderDecoderRemovalNetworks.get_autoencoder(img_size=self.img_size)

        # config
        self.EPOCH = 100
        self.inc = 0
        self.save_every = 10
        self.output_every = 2

    def save_weights(self):
        self.rm.save_weights('../save/' + 'encoderDecoderRm_' + str(self.inc) + '.h5')

    def output_middle_result(self, rows=5):
        iter = self.val_dataset.__iter__()
        img_lists = []
        for _ in range(rows):
            img_list = []
            t, r, m = next(iter)
            t1 = tf.squeeze(t, axis=0)
            r1 = tf.squeeze(r, axis=0)
            m1 = tf.squeeze(m, axis=0)
            img_list.append(t1)
            img_list.append(r1)
            img_list.append(m1)

            # forward
            pred_t = self.rm(m)
            pred_t = tf.squeeze(pred_t, axis=0)
            img_list.append(pred_t)
            img_lists.append(img_list)

        ImageUtils.plot_images(rows, 3 + 1, img_lists, is_save=True, epoch_index=self.inc)

    @tf.function
    def train_one_step(self, t, r, m):
        with tf.GradientTape() as tape:
            pred_t = self.rm(m)
            loss = self.compute_l2_loss(pred_t, t) + self.compute_feature_loss(pred_t, t)
            grad = tape.gradient(loss, self.rm.trainable_variables)
            self.g_optimizer.apply_gradients(zip(grad, self.rm.trainable_variables))

    def start_train_task(self):
        for _ in range(self.EPOCH):
            self.inc += 1
            for t, r, m in self.train_dataset:
                self.train_one_step(t, r, m)
                if self.inc % self.save_every == 0:
                    self.save_weights()

                if self.inc % self.output_every == 0:
                    self.output_middle_result()

    def compute_l2_loss(self, img1, img2):
        return tf.reduce_mean((img1 - img2) ** 2)

    def compute_feature_loss(self, img1, img2):
        """
        compute the feature using vgg19
        :param img1: image1
        :param img2: image2
        :return: loss tensor
        """
        feat_list_img1 = self.feature_extractor(img1)
        feat_list_img2 = self.feature_extractor(img2)
        feature_loss = 0
        for i in range(len(feat_list_img1)):
            feature_loss += tf.reduce_mean(tf.abs(feat_list_img1[i] - feat_list_img2[i]))

        return feature_loss


class BeyondLinearityRemovalModel:
    def __init__(self):
        # epsilon for log function
        self.EPS = 1e-12

        # the syn net, output is a tensor list of [mask, synthetic image]
        self.syn = BeyondLinearityNetworks.build_SynNet()

        # the rm net, output is a tensor list of [t_layer, r_layer, recombined_image].
        self.rm = BeyondLinearityNetworks.build_RmNet()

        # optimizer for g and d
        self.syn_optimizer = keras.optimizers.Adam(learning_rate=0.0002)
        self.rm_optimizer = keras.optimizers.Adam(learning_rate=0.0002)
        self.d_optimizer = keras.optimizers.Adam(learning_rate=0.0002)

        # the vgg19 feature extractor
        self.feature_extractor = Vgg19FeatureExtractor.build_vgg19_feature_extractor()

        # syn discriminator, disc the reflective images and normal images
        self.syn_d = BeyondLinearityNetworks.build_discriminator()

        # rm discriminator, disc the non-reflective images and normal images.
        self.rm_d = BeyondLinearityNetworks.build_discriminator()

    def compute_smoothness_loss(self, mask):
        """
        compute the smoothness_loss
        :param mask:
        :return:
        """
        grad_x = mask[:, 1:, :, :] - mask[:, :-1, :, :]
        grad_y = mask[:, :, 1:, :] - mask[:, :, :-1, :]

        grad_loss = tf.reduce_mean(grad_x) + tf.reduce_mean(grad_y)

        return grad_loss

    @tf.function
    def train_one_step(self, t, r, m):
        # training syn_D and rm_D
        cat_tr = tf.concat([t, r], axis=3)
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2, tf.GradientTape() as tape3:
            # train D
            mask, syn_m = self.syn(cat_tr, training=True)
            on_real = tf.reduce_mean(self.syn_d(m))
            on_fake = tf.reduce_mean(self.syn_d(syn_m))

            loss_d_syn = tf.reduce_mean(tf.math.log(on_real + self.EPS) + tf.math.log(1 - on_fake + self.EPS))
            grad_d_syn = tape1.gradient(loss_d_syn, self.syn_d.trainable_variables)
            self.d_optimizer.apply_gradients(zip(grad_d_syn, self.syn_d.trainable_variables))

            # train syn net
            smoothness_loss = self.compute_smoothness_loss(mask)
            gan_loss = tf.reduce_mean(-tf.math.log(self.syn_d(syn_m) + self.EPS))

            Loss_syn = 10 * smoothness_loss + gan_loss
            grad_syn = tape2.gradient(Loss_syn, self.syn.trainable_variables)
            self.syn_optimizer.apply_gradients(zip(grad_syn, self.syn.trainable_variables))

            # todo




