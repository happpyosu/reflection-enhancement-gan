import tensorflow as tf
from Network.network import Network, PerceptionRemovalNetworks
from Dataset.dataset import DatasetFactory

'''
This file offers some reflection removal model implements.
(1) PerceptionRemovalModel: reflection removal with the perception loss
@author: chen hao
@date: 2020-10-31
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

        # epsilon for log function
        self.EPS = 1e-12

        # the image size
        self.img_size = 256

        # the vgg19 feature extractor
        self.feature_extractor = self._build_vgg19()

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

            loss_d = 0.5 * tf.reduce_mean(-tf.math.log(on_real + self.EPS + tf.math.log(1 - on_fake + self.EPS)))
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

    def _build_vgg19(self):
        """
        build the VGG19 submodel used for building perceptual middle features.
        :return: tf.Model
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


if __name__ == '__main__':
    rm = PerceptionRemovalModel()
    ite = rm.train_dataset.__iter__()
    t, r, m = next(ite)

    rm.train_one_step(t, r, m)
