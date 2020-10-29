import tensorflow as tf
from Network.network import Network
from Dataset.dataset import DatasetFactory
from tensorflow.keras.losses import MSE


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

        # models
        self.D1 = Network.build_discriminator(img_size=self.img_size)
        self.D2 = Network.build_discriminator(img_size=self.img_size)
        self.G = Network.build_generator(img_size=self.img_size, noise_dim=self.noise_dim)
        self.E = Network.build_encoder(img_size=self.img_size, noise_dim=self.noise_dim)

        # optimizer
        self.optimizer_D1 = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.5)
        self.optimizer_D2 = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.5)
        self.optimizer_G = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.5)
        self.optimizer_E = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.5)

        # lr decay
        self.lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=2e-4,
                                                                          decay_steps=100,
                                                                          decay_rate=0.5)

        # dataset
        self.train_dataset = DatasetFactory.get_dataset_by_name(name="RealDataset", mode="train")
        self.val_dataset = DatasetFactory.get_dataset_by_name(name="RealDataset", mode='val')

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
        with tf.GradientTape() as d1_tape, tf.GradientTape() as d2_tape, tf.GradientTape() as G_tape, tf.GradientTape() as E_tape:
            # step 1. ----------------------Train D1----------------------
            cat_trm = tf.concat([t1, r1, m1], axis=3)
            mu, log_var = self.E(cat_trm, training=True)
            std = tf.math.exp(log_var / 2)

            # encode the noise vector and tile to an image
            z = tf.random.normal(shape=(1, 1, 1, self.noise_dim))
            z = (z * std) + mu
            z = tf.tile(z, [1, self.img_size, self.img_size, 1])

            cat_trn = tf.concat([t1, r1, z], axis=3)
            fake_m_VAE = self.G(cat_trn, training=True)
            real_score1, real_score2 = self.D1(m1, training=True)
            fake_score1, fake_score2 = self.D1(fake_m_VAE, training=True)

            D1_loss1 = tf.reduce_mean((real_score1 - tf.ones_like(real_score1)) ** 2, keepdims=True) + tf.reduce_mean(
                (fake_score1 - tf.zeros_like(fake_score1)) ** 2, keepdims=True)
            D1_loss2 = tf.reduce_mean((real_score2 - tf.ones_like(real_score2)) ** 2, keepdims=True) + tf.reduce_mean(
                (fake_score2 - tf.zeros_like(fake_score2)) ** 2, keepdims=True)

            # D1 total Loss.
            D1_Loss = D1_loss1 + D1_loss2

            # step 2. ----------------------Train D2----------------------
            # random z directly sampled from normal distribution
            z = tf.random.normal(shape=(1, 1, 1, self.noise_dim))
            z = tf.tile(z, [1, self.img_size, self.img_size, 1])

            # generate fake images
            cat_trn = tf.concat([t2, r2, z], axis=3)
            fake_m_LR = self.G(cat_trn, training=True)

            # get score on real and fake images
            real_score1, real_score2 = self.D2(m2)
            fake_score1, fake_score2 = self.D2(fake_m_LR)








