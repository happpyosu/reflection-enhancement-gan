import sys
sys.path.append('../')
from Network.network import InfoGNetworks
import tensorflow as tf
from Dataset.dataset import CategoricalReflectionDataset
import numpy as np
from utils.imageUtils import ImageUtils
from utils import gpuutils


class ReflectionInfoG:
    def __init__(self):
        # context
        self.img_size = 256

        # latent code dims
        self.latent_dim = 10

        # class num, refers to the reflection types, such as the blurring, ghosting...
        self.n_class = 2

        # code dims
        self.code_dim = 4

        # the generator
        self.G = InfoGNetworks.build_Generator(img_size=self.img_size,
                                               input_dim=3 + 3 + self.latent_dim + self.n_class + self.code_dim)

        # the discriminator
        self.D = InfoGNetworks.build_Discriminator(self.img_size, self.n_class, self.code_dim)

        # Categorical Cross entropy loss function
        self.cce = tf.keras.losses.categorical_crossentropy

        # lr
        lr_schedule = 0.0002

        # optimizer for D and G
        self.optimizer_D = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)
        self.optimizer_G = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)

        # config logging
        self.inc = 0
        self.save_every = 10
        self.output_every = 2
        self.EPOCH = 100

        # dataset
        self.dataset = CategoricalReflectionDataset(step_per_epoch=3000)

        # testset
        self.valSet = CategoricalReflectionDataset(step_per_epoch=1000)

    def gen_normal_noise(self):
        # generate a noise obey gaussian distribution
        z = tf.random.normal(shape=(1, 1, 1, self.latent_dim))
        z = tf.tile(z, [1, self.img_size, self.img_size, 1])
        return z

    def save_weights(self):
        self.G.save_weights('../save/' + 'info_G_' + str(self.inc) + '.h5')

    def output_middle_results(self, rows=5, cols=5):
        iter = self.valSet.__iter__()
        img_lists = []
        for _ in range(rows):
            img_list = []
            t, r, m, c = next(iter)
            img_list.append(tf.squeeze(t, axis=0))
            img_list.append(tf.squeeze(r, axis=0))
            img_list.append(tf.squeeze(m, axis=0))
            for _ in range(cols):
                z0 = self.gen_normal_noise()
                z1 = self.gen_uniform_noise()
                z2 = self.gen_class(self.gen_random_onehot())
                inp_G = tf.concat([t, r, z0, z1, z2], axis=3)
                out_G = tf.squeeze(self.G(inp_G), axis=0)
                img_list.append(out_G)
            img_lists.append(img_list)

        ImageUtils.plot_images(5, 3+cols, img_lists, is_save=True, epoch_index=self.inc)

    def gen_random_onehot(self):
        rand_c = np.random.randint(0, self.n_class)
        return tf.one_hot([rand_c], depth=self.n_class)

    def gen_uniform_noise(self):
        # generate a noise obey uniform distribution
        z = tf.random.uniform(shape=(1, 1, 1, self.code_dim), minval=-1, maxval=1)
        z = tf.tile(z, [1, self.img_size, self.img_size, 1])
        return z

    def gen_class(self, one_hot_z):
        # generate the class info for concat
        z = tf.expand_dims(tf.expand_dims(one_hot_z, axis=0), axis=0)
        z = tf.tile(z, [1, self.img_size, self.img_size, 1])
        return z

    def start_train_task(self):
        for _ in range(self.EPOCH):
            self.inc += 1
            for t, r, m, c in self.dataset:
                self.train_one_step(t, r, m, c)

            if self.inc % self.save_every == 0:
                self.save_weights()

            if self.inc % self.output_every == 0:
                self.output_middle_results()

    @tf.function
    def train_one_step(self, t, r, m, c):
        """
        train info-G a step
        :param t: transmission layer.
        :param r: reflection layer.
        :param m: mixture layer.
        :param c: class in one-hot coding.
        :return:
        """
        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            z0 = self.gen_normal_noise()
            z1 = self.gen_uniform_noise()
            z2 = self.gen_class(c)

            # ----------
            #  train G
            # ----------
            g_inp = tf.concat([t, r, z0, z1, z2], axis=3)
            gen_img = self.G(g_inp)

            d_inp = tf.concat([t, r, gen_img], axis=3)
            validity, _, _ = self.D(d_inp)
            g_loss = tf.reduce_mean((validity - tf.ones_like(validity)) ** 2) + 2 * tf.reduce_mean(tf.abs(gen_img - m))

            g_grad = G_tape.gradient(g_loss, self.G.trainable_variables)
            self.optimizer_G.apply_gradients(zip(g_grad, self.G.trainable_variables))

            # ----------
            #  train D
            # ----------
            real_inp = tf.concat([t, r, m], axis=3)
            fake_inp = tf.concat([t, r, gen_img], axis=3)
            real_pred, _, _ = self.D(real_inp)
            fake_pred, _, _ = self.D(fake_inp)
            d_real_loss = tf.reduce_mean((real_pred - tf.ones_like(real_pred)) ** 2)
            d_fake_loss = tf.reduce_mean((fake_pred - tf.zeros_like(fake_pred)) ** 2)

            d_loss = (d_real_loss + d_fake_loss) / 2

            d_grad = D_tape.gradient(d_loss, self.D.trainable_variables)
            self.optimizer_D.apply_gradients(zip(d_grad, self.D.trainable_variables))

        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            # ----------
            #    info
            # ----------
            gt_label = c
            z0 = self.gen_normal_noise()
            z1 = self.gen_uniform_noise()
            z2 = self.gen_class(c)
            g_inp = tf.concat([t, r, z0, z1, z2], axis=3)

            gen_img = self.G(g_inp)
            fake_inp = tf.concat([t, r, gen_img], axis=3)
            _, pred_label, pred_code = self.D(fake_inp)

            info_loss = 1.5 * self.cce(gt_label, pred_label, from_logits=True) + 0.1 * tf.reduce_mean((z1 - pred_code) ** 2)
            g_grad = G_tape.gradient(info_loss, self.G.trainable_variables)
            d_grad = D_tape.gradient(info_loss, self.D.trainable_variables)

            self.optimizer_G.apply_gradients(zip(g_grad, self.G.trainable_variables))
            self.optimizer_D.apply_gradients(zip(d_grad, self.D.trainable_variables))

        # with tf.GradientTape() as G_tape:
        #     z0 = self.gen_normal_noise()
        #     z1 = self.gen_uniform_noise()
        #     z2 = self.gen_class(c)


if __name__ == '__main__':
    gpuutils.which_gpu_to_use(0)
    R = ReflectionInfoG()
    R.start_train_task()



