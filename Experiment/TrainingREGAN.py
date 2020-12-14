import tensorflow as tf
import sys

sys.path.append('../')
from Network.network import Network
from Dataset.dataset import DatasetFactory
from utils.imageUtils import ImageUtils
from utils import gpuutils


class ReflectionGAN:
    """
        @author: chen hao
        @Date:   2020.5.7
        ReflectionGAN: a model for generating multimodal reflective images. The model includes a generator G, an
        Encoder E and two discriminators D1 and D2. To avoid the mode-collapse problem in conditional generation,
        an cross domain generation strategy is used. To understand the main idea, we define the following three
        domains and two process:

        domains:
        Image triplet (T, R, S) ......................................................................(domain A)
        one-dimensional latent code (z) ..............................................................(domain B)
        generated reflective image S' which is faithful to (T, R) ....................................(domain C)

        processes:
        #1 forward translation  [ (T, R, S) ----E----> z ] and [(T, R, z)  ----G----> S']
        #2 backward translation [ (T, R, z) ----G----> S'] and [(T, R, S') ----E----> z']
    """

    def __init__(self):
        # config
        self.noise_dim = 8
        self.img_size = 256
        self.epoch = 100

        # models
        self.D1 = Network.build_discriminator(img_size=self.img_size)
        self.D2 = Network.build_discriminator(img_size=self.img_size)
        self.G = Network.build_generator(img_size=self.img_size, noise_dim=self.noise_dim)
        self.E = Network.build_encoder(img_size=self.img_size, noise_dim=self.noise_dim)

        # dataset
        self.train_dataset = DatasetFactory.get_dataset_by_name(name="RealDataset", mode="train")
        self.val_dataset = DatasetFactory.get_dataset_by_name(name="RealDataset", mode='val')

        # lr decay
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.001,
                                                                     decay_steps=self.epoch * 5000 // 2,
                                                                     decay_rate=1,
                                                                     staircase=False)
        lr_schedule = 0.0002

        # optimizer
        self.optimizer_D1 = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)
        self.optimizer_D2 = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)
        self.optimizer_G = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)
        self.optimizer_E = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)

        # config logging
        self.inc = 0
        self.save_every = 10
        self.output_every = 2

    def _gen_noise(self):
        # generate a noise
        z = tf.random.normal(shape=(1, 1, 1, self.noise_dim))
        z = tf.tile(z, [1, self.img_size, self.img_size, 1])
        return z

    def output_middle_result(self, rows=5, cols=5):
        # get a test batch
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
            for _ in range(cols):
                z = self._gen_noise()
                trn_concat = tf.concat([t, r, z], axis=3)
                out = self.G(trn_concat)
                out = tf.squeeze(out, axis=0)
                img_list.append(out)
            img_lists.append(img_list)

        ImageUtils.plot_images(rows, cols + 3, img_lists, is_save=True, epoch_index=self.inc)

    def start_train_task(self):
        self.inc = 0
        for _ in range(self.epoch):
            self.inc += 1
            print('[info]: current epoch: ' + str(self.inc))
            for (t, r, m) in self.train_dataset:
                self.train_one_step(t, r, m)

            if self.inc % self.save_every == 0:
                self.save_weights()

            if self.inc % self.output_every == 0:
                self.output_middle_result()
        # save the final weight
        self.save_weights()

    def save_weights(self):
        self.G.save_weights('../save/' + 'G_' + str(self.inc) + '.h5')
        self.E.save_weights('../save/' + 'E_' + str(self.inc) + '.h5')

    def load_weights(self, epoch: int):
        self.G.load_weights('../save/' + 'G_' + str(epoch) + '.h5')
        # self.E.load_weights('../save/' + 'E_' + str(epoch) + '.h5')

    @tf.function
    def train_one_step(self, t, r, m):
        """
        Train the whole model a step.
        :param t: transmission layer
        :param r: reflection layer
        :param m: dirty images
        :return: None
        """
        t1, t2 = tf.split(t, 2, 0)
        r1, r2 = tf.split(r, 2, 0)
        m1, m2 = tf.split(m, 2, 0)

        # some concat images.
        cat_tr_VAE = tf.concat([t1, r1], axis=3)
        cat_tr_LR = tf.concat([t2, r2], axis=3)
        cat_trm_VAE = tf.concat([cat_tr_VAE, m1], axis=3)

        with tf.GradientTape() as d1_tape, \
                tf.GradientTape() as d2_tape:

            # step 1. ----------------------Train D1----------------------
            mu, log_var = self.E(cat_trm_VAE, training=True)
            std = tf.exp(log_var / 2)

            # encode the noise vector and tile to an image
            z = tf.random.normal(shape=(1, 1, 1, self.noise_dim))
            z = (z * std) + mu
            z = tf.tile(z, [1, self.img_size, self.img_size, 1])

            cat_trn_VAE = tf.concat([cat_tr_VAE, z], axis=3)
            fake_m_VAE = self.G(cat_trn_VAE, training=True)

            # D1 outputs 1 if fed with real image, else 0
            real_score1, real_score2 = self.D1(m1, training=True)
            fake_score1, fake_score2 = self.D1(fake_m_VAE, training=True)
            D1_loss1 = tf.reduce_mean((real_score1 - tf.ones_like(real_score1)) ** 2) + tf.reduce_mean(
                (fake_score1 - tf.zeros_like(fake_score1)) ** 2)
            D1_loss2 = tf.reduce_mean((real_score2 - tf.ones_like(real_score2)) ** 2) + tf.reduce_mean(
                (fake_score2 - tf.zeros_like(fake_score2)) ** 2)

            # D1 total Loss.
            D1_Loss = D1_loss1 + D1_loss2
            print("D1 Loss:", str(D1_Loss))
            # step 2. ----------------------Train D2----------------------
            # random z directly sampled from normal distribution
            z = tf.random.normal(shape=(1, 1, 1, self.noise_dim))
            z = tf.tile(z, [1, self.img_size, self.img_size, 1])

            # generate fake images
            cat_trn_LR = tf.concat([cat_tr_LR, z], axis=3)
            fake_m_LR = self.G(cat_trn_LR, training=True)

            # get score on real and fake images
            real_score1, real_score2 = self.D2(m2)
            fake_score1, fake_score2 = self.D2(fake_m_LR)

            D2_loss1 = tf.reduce_mean((real_score1 - tf.ones_like(real_score1)) ** 2) + tf.reduce_mean(
                (fake_score1 - tf.zeros_like(fake_score1)) ** 2)
            D2_loss2 = tf.reduce_mean((real_score2 - tf.ones_like(real_score2)) ** 2) + tf.reduce_mean(
                (fake_score2 - tf.zeros_like(fake_score2)) ** 2)

            # D2 total loss
            D2_Loss = D2_loss1 + D2_loss2

            print("D2 Loss:", str(D2_Loss))

            # update D1 and D2
            grad_D1 = d1_tape.gradient(D1_Loss, self.D1.trainable_variables)
            grad_D2 = d2_tape.gradient(D2_Loss, self.D2.trainable_variables)
            self.optimizer_D1.apply_gradients(zip(grad_D1, self.D1.trainable_variables))
            self.optimizer_D2.apply_gradients(zip(grad_D2, self.D2.trainable_variables))

        with tf.GradientTape() as G_tape, \
                tf.GradientTape() as G_tape1, \
                tf.GradientTape() as E_tape:
            # step 3. ---------------------- Train G to fool the discriminators ----------------------
            # get the encoded z from Encoder.
            mu, var = self.E(cat_trm_VAE)
            std = tf.exp(var / 2)
            z = tf.random.normal(shape=(1, 1, 1, self.noise_dim))
            z = (z * std) + mu
            z = tf.tile(z, [1, self.img_size, self.img_size, 1])

            cat_trn_VAE = tf.concat([cat_tr_VAE, z], axis=3)
            fake_m_VAE = self.G(cat_trn_VAE, training=True)
            fake_D1_1, fake_D1_2 = self.D1(fake_m_VAE)
            GAN_loss_cVAE_1 = tf.reduce_mean((fake_D1_1 - tf.ones_like(fake_D1_1)) ** 2)
            GAN_loss_cVAE_2 = tf.reduce_mean((fake_D1_2 - tf.ones_like(fake_D1_2)) ** 2)

            # get an prior noise from gaussian distribution.
            random_z = tf.random.normal(shape=(1, 1, 1, self.noise_dim))
            z = tf.tile(random_z, [1, self.img_size, self.img_size, 1])

            cat_trn_LR = tf.concat([cat_tr_LR, z], axis=3)
            fake_m_LR = self.G(cat_trn_LR, training=True)
            fake_D2_1, fake_D2_2 = self.D2(fake_m_LR)

            GAN_loss_cLR_1 = tf.reduce_mean((fake_D2_1 - tf.ones_like(fake_D2_1)) ** 2)
            GAN_loss_cLR_2 = tf.reduce_mean((fake_D2_2 - tf.ones_like(fake_D2_2)) ** 2)

            # gan loss for G
            G_GAN_Loss = GAN_loss_cVAE_1 + GAN_loss_cVAE_2 + GAN_loss_cLR_1 + GAN_loss_cLR_2

            print('G_GAN_loss', str(G_GAN_Loss))

            # KL-divergence loss for G and E
            KL_div = 0.01 * tf.reduce_sum(0.5 * (mu ** 2 + tf.exp(var) - var - 1))

            # step. 4. Reconstruct of ground truth image
            img_recon_loss = 10 * tf.reduce_mean(tf.abs(fake_m_VAE - m1))

            # total loss for Encoder and Generator
            E_G_Loss = G_GAN_Loss + KL_div + img_recon_loss

            print('KL_loss', str(KL_div))
            print('img_recon_loss', str(img_recon_loss))

            # update E and G
            E_grad = E_tape.gradient(E_G_Loss, self.E.trainable_variables)
            G_grad = G_tape.gradient(E_G_Loss, self.G.trainable_variables)
            self.optimizer_E.apply_gradients(zip(E_grad, self.E.trainable_variables))
            self.optimizer_G.apply_gradients(zip(G_grad, self.G.trainable_variables))

            # step 5. ---------------------- Train G to reconstruct the latent code z ----------------------
            cat_trm_fake = tf.concat([cat_tr_LR, fake_m_LR], axis=3)
            mu_, var_ = self.E(cat_trm_fake, training=True)
            z_recon_Loss = 0.5 * tf.reduce_mean(tf.abs(random_z - mu_))

            # update G
            grad_G = G_tape1.gradient(z_recon_Loss, self.G.trainable_variables)
            self.optimizer_G.apply_gradients(zip(grad_G, self.G.trainable_variables))
            print('z_recon_Loss', str(z_recon_Loss))
            # step 6. ---------------------- Mode seeking term (optional) -----------------------
            # Do nothing
            # ----debug----
            print('G weight: ', self.G.trainable_variables)


if __name__ == '__main__':
    # gpuutils.which_gpu_to_use(1)
    gan = ReflectionGAN()
    gan.load_weights(20)
    gan.output_middle_result(4, 8)
    # gan.start_train_task()
