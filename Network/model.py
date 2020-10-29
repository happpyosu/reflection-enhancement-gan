import tensorflow as tf
from Network.network import Network
from Dataset.dataset import DatasetFactory


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
        self.noise_dim = 4
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

    @tf.function
    def train_D(self, t, r, m, which_D=1):
        """
            train Discriminator a step.
            :param which_D: train D1 or D2, which can either be 1 or 2
            :param t: transmission layer image (T), which is packed to a 4-D tensor (1, img_size, img_size, 3)
            :param r: reflection layer image (R), which is packed to a 4-D tensor (1, img_size, img_size, 3)
            :param m: mixture image (real S), which is packed to a 4-D tensor (1, img_size, img_size, 3)
            :return: None
            :raise: ValueError, if the which_D parameter is not valid.
        """

        if which_D == 1:
            D = self.D1
        elif which_D == 2:
            D = self.D2
        else:
            raise ValueError("Unexpected discriminator index, possible value: 1 or 2, but got" + str(which_D))

        with tf.GradientTape() as D_tape:
            cat_trm = tf.concat([t, r, m], axis=3)
            mu, log_var = self.E(cat_trm, training=True)
            std = tf.math.exp(log_var / 2)

            # tile the one-dimensional noise vector to an two-dimensional noise matrix
            z = tf.random.normal(shape=(self.noise_dim,))
            z = (z * std[0]) + mu[0]
            z = z[tf.newaxis, tf.newaxis, tf.newaxis, :]
            z = tf.tile(z, [1, self.img_size, self.img_size, 1])

            cat_trn = tf.concat([t, r, z], axis=3)
            fake_m = self.G(cat_trn, training=True)
            real_score1, real_score2 = D(m, training=True)
            fake_score1, fake_score2 = D(fake_m, training=True)

            loss1 = tf.reduce_mean((real_score1 - tf.ones_like(real_score1)) ** 2) + tf.reduce_mean((fake_score1 - tf.zeros_like(fake_score1)) ** 2)
            loss2 = tf.reduce_mean((real_score2 - tf.ones_like(real_score2)) ** 2) + tf.reduce_mean((fake_score2 - tf.zeros_like(fake_score2)) ** 2)

            Loss = loss1 + loss2

            grad_D = D_tape.gradient(Loss, D.trainable_variables)
            self.optimizer_D1.apply_gradients(zip(grad_D, D.trainable_variables))
           # todo
    @tf.function
    def forward_translation_training(self, t1, r1, m1):
        with tf.GradientTape() as G_tape, tf.GradientTape() as E_tape:
            # ---------------------------Section one-------------------------------
            # Section one  (t1, r1, m1) --E--> (z1), (t1, r1, z1) --G--> fake_m1
            # please refer to Fig.3(b) in the paper
            cat_trm = tf.concat([t1, r1, m1], axis=3)
            mu, log_var = self.E(cat_trm, training=True)
            std = tf.math.exp(log_var / 2)

            z1 = tf.random.normal(shape=(self.noise_dim,))
            z1 = (z1 * std[0]) + mu[0]
            z1 = z1[tf.newaxis, tf.newaxis, tf.newaxis, :]
            z1 = tf.tile(z1, [1, self.img_size, self.img_size, 1])

            cat_trn1 = tf.concat([t1, r1, z1], axis=3)
            fake_m1 = self.G(cat_trn1, training=True)
            D1_fake_score1, D1_fake_score2 = self.D1(fake_m1, training=True)
            D1_loss1 = tf.reduce_mean((D1_fake_score1 - tf.ones_like(D1_fake_score1)) ** 2)
            D1_loss2 = tf.reduce_mean((D1_fake_score2 - tf.ones_like(D1_fake_score2)) ** 2)
            recon_loss = 10 * tf.reduce_mean(tf.abs(m1 - fake_m1))
            # ------------------------end of section one---------------------------

            # ---------------------------Section two-------------------------------
            # Section two. KL(N(z), q(z)) should be minimized
            # please refer to Fig.3(b) (medium part) in the paper.
            KL_div_loss = 0.01 * tf.reduce_sum(0.5 * (mu ** 2 + tf.exp(log_var) - log_var - 1))
            # ------------------------end of section two---------------------------

            # update E and G
            final_loss = D1_loss1 + D1_loss2 + recon_loss + KL_div_loss
            grad_E = E_tape.gradient(final_loss, self.E.trainable_variables)
            grad_G = G_tape.gradient(final_loss, self.G.trainable_variables)
            self.optimizer_E.apply_gradients(zip(grad_E, self.E.trainable_variables))
            self.optimizer_G.apply_gradients(zip(grad_G, self.G.trainable_variables))

    def backward_translation_training(self, t2, r2):
        with tf.GradientTape() as G_tape, tf.GradientTape() as E_tape:
            # ---------------------------Section one-----------------------------
            # Section three. (t2, r2, z2) --G--> fake_m2
            # please refer to Fig.3(c) (left part) in the paper.
            z2 = tf.random.normal(shape=(self.noise_dim,))
            z2 = z2[tf.newaxis, tf.newaxis, tf.newaxis, :]
            z2 = tf.tile(z2, [1, self.img_size, self.img_size, 1])
            cat_trn2 = tf.concat([t2, r2, z2], axis=3)
            fake_m2 = self.G(cat_trn2, training=True)
            D2_fake_score1, D2_fake_score2 = self.D2(fake_m2)
            D2_loss1 = tf.reduce_mean((D2_fake_score1 - tf.ones_like(D2_fake_score1)) ** 2)
            D2_loss2 = tf.reduce_mean((D2_fake_score2 - tf.ones_like(D2_fake_score2)) ** 2)
            # ---------------------------end of Section one----------------------

            # ---------------------------Section two-----------------------------
            # Section Four. (t2, r2, fake_m2) --E--> z2'
            # please refer to Fig.3(c) (right part) in the paper.
            cat_tr_fake = tf.concat([t2, r2, fake_m2], axis=3)
            mu, log_var = self.E(cat_tr_fake, training=True)
            recon_loss = tf.reduce_mean(tf.abs(z2 - mu))

            final_G_loss = D2_loss1 + D2_loss2 + recon_loss
            final_E_loss = D2_loss1 + D2_loss1

            grad_E = E_tape.gradient(final_E_loss, self.E.trainable_variables)
            grad_G = G_tape.gradient(final_G_loss, self.G.trainable_variables)

            self.optimizer_E.apply_gradients(zip(grad_E, self.E.trainable_variables))
            self.optimizer_G.apply_gradients(zip(grad_G, self.G.trainable_variables))











if __name__ == '__main__':
    gan = ReflectionGAN()
    for t1, r1, m1 in gan.train_dataset:
        gan.train_D(t1, r1, m1, 1)



