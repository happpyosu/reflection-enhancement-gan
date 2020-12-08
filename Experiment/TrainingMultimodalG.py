import tensorflow as tf
import sys

sys.path.append('../')
from Network.network import Network, Vgg19FeatureExtractor
from Dataset.dataset import DatasetFactory
from utils.imageUtils import ImageUtils
from utils import gpuutils


class Image2Reflection:
    def __init__(self):
        # config
        self.noise_dim = 4
        self.img_size = 256
        self.epoch = 100
        self.EPS = 0.0000001

        # default noise dim is 4
        self.G = Network.build_multimodal_G(noise_dim=self.noise_dim)
        self.D = Network.build_multimodal_D(img_size=self.img_size)

        # dataset
        self.train_dataset = DatasetFactory.get_dataset_by_name(name="SynDataset", mode="train")
        self.val_dataset = DatasetFactory.get_dataset_by_name(name="SynDataset", mode="val")

        # lr
        lr_schedule = 0.0002

        # optimizer for D and G
        self.optimizer_D = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)
        self.optimizer_G = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)

        # config logging
        self.inc = 0
        self.save_every = 10
        self.output_every = 2

        # build vgg19 feature extractor
        self.vgg19 = Vgg19FeatureExtractor.build_vgg19_feature_extractor()

    def _gen_noise(self):
        # generate a noise
        z = tf.random.normal(shape=(1, 1, 1, self.noise_dim))
        z = tf.tile(z, [1, self.img_size, self.img_size, 1])
        return z

    def save_weights(self):
        self.G.save_weights('../save/' + 'G_multi_' + str(self.inc) + '.h5')

    def load_weights(self, epoch: int):
        self.G.load_weights('../save/' + 'G_multi_' + str(epoch) + '.h5')

    def output_middle_result(self, rows=5, cols=5):
        # get a test batch
        iter = self.val_dataset.__iter__()
        img_lists = []
        for _ in range(rows):
            img_list = []
            _, r, rb, _ = next(iter)
            r1 = tf.squeeze(r, axis=0)
            rb1 = tf.squeeze(rb, axis=0)
            img_list.append(r1)
            img_list.append(rb1)
            for _ in range(cols):
                z = self._gen_noise()
                r_with_noise = tf.concat([r, z], axis=3)
                out = self.G(r_with_noise)
                out = tf.squeeze(out, axis=0)
                img_list.append(out)
            img_lists.append(img_list)

        ImageUtils.plot_images(rows, cols + 2, img_lists, is_save=True, epoch_index=self.inc)

    def start_train_task(self):
        self.inc = 0
        for _ in range(self.epoch):
            self.inc += 1
            print('[info]: current epoch: ' + str(self.inc))
            for _, r, rb, _ in self.train_dataset:
                self.train_one_step(r, rb)

            if self.inc % self.save_every == 0:
                self.save_weights()

            if self.inc % self.output_every == 0:
                self.output_middle_result()
        # save the final weight
        self.save_weights()

    def l1_distance(self, x, y):
        """
        interface for calculating the l1 loss
        :param x:
        :param y:
        :return:
        """
        return tf.reduce_mean(tf.abs(x - y))

    def compute_perceptual_loss(self, img1, img2):
        f1 = self.vgg19(img1)
        f2 = self.vgg19(img2)

        # l1 loss
        loss = self.l1_distance(img1, img2)

        # perceptual loss
        for fe1, fe2 in zip(f1, f2):
            loss += self.l1_distance(fe1, fe2)

        return loss

    @tf.function
    def train_one_step(self, r, rb):
        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            # train D
            noise = self._gen_noise()

            c_with_real = tf.concat([r, rb], axis=3)
            r_with_noise = tf.concat([r, noise], axis=3)
            fake_rb = self.G(r_with_noise)

            c_with_fake = tf.concat([r, fake_rb], axis=3)

            on_fake1, on_fake2 = self.D(c_with_fake)
            on_real1, on_real2 = self.D(c_with_real)

            D_loss1 = tf.reduce_mean((on_real1 - tf.ones_like(on_real1)) ** 2) + \
                      tf.reduce_mean((on_fake1 - tf.zeros_like(on_fake1)) ** 2)

            D_loss2 = tf.reduce_mean((on_real2 - tf.ones_like(on_real2)) ** 2) + \
                      tf.reduce_mean((on_fake2 - tf.zeros_like(on_fake2)) ** 2)

            D_loss = D_loss1 + D_loss2

            grad_D = D_tape.gradient(D_loss, self.D.trainable_variables)
            self.optimizer_D.apply_gradients(zip(grad_D, self.D.trainable_variables))

            # train G
            noise = self._gen_noise()
            r_with_noise = tf.concat([r, noise], axis=3)
            fake_rb = self.G(r_with_noise)

            c_with_fake = tf.concat([r, fake_rb], axis=3)

            # l1 loss
            # l1_loss = 10 * tf.reduce_mean(tf.abs(fake_rb - rb))
            # replaced with the perceptual loss
            l1_loss = 10 * self.compute_perceptual_loss(fake_rb, rb)

            # gan loss
            on_fake1, on_fake2 = self.D(c_with_fake)
            gan_loss = tf.reduce_mean((on_fake1 - tf.ones_like(on_fake1)) ** 2) + \
                       tf.reduce_mean((on_fake2 - tf.ones_like(on_fake1)) ** 2)

            # modal seeking loss
            noise2 = self._gen_noise()
            r_with_noise2 = tf.concat([r, noise2], axis=3)
            fake_rb2 = self.G(r_with_noise2)

            modal_seeking_loss = tf.reduce_sum(tf.abs(noise2 - noise)) / (self.compute_perceptual_loss(fake_rb2, fake_rb) + self.EPS)

            G_loss = l1_loss + 0.01 * gan_loss + 0.01 * modal_seeking_loss

            grad_G = G_tape.gradient(G_loss, self.G.trainable_variables)
            self.optimizer_G.apply_gradients(zip(grad_G, self.G.trainable_variables))


if __name__ == '__main__':
    gpuutils.which_gpu_to_use(2)
    gan = Image2Reflection()
    # gan.load_weights(100)
    # gan.output_middle_result(4, 5)
    gan.start_train_task()