from Experiment.TrainingREGAN import ReflectionGAN
from utils.imageUtils import ImageUtils
from Dataset.dataset import DatasetFactory
import tensorflow as tf


class ReflectionGANTest:
    def __init__(self, which_epoch):
        # init the reflection GAN
        self.gan = ReflectionGAN()

        # load epoch weights
        self.gan.load_weights(epoch=which_epoch)

        # test set
        # self.testSet = DatasetFactory.get_dataset_by_name("TestDataset")

    def modal_transfer(self, idx1, idx2):
        """
        Test the modal transfer function. In the dataset, assume we have two sets of image pair
        (t1, r1, m1) and (t2, r2, m2), the modal transfer test is done as the following steps:
        (1) extract modal (t1, r1, m1) ----E----> z1, (t2, r2, m2) ----E----> z2
        (2) transfer (t1, r1, z2) ----G----> m1', (t2, r2, m2) ----G----> m2'
        (3) compare m1 with m1' and m2 with m2'
        :param idx1: image index1 in val set
        :param idx2: image index2 in val set
        :return: None
        """
        img_lists = []
        img_list1 = []
        img_list2 = []

        testSet = DatasetFactory.get_dataset_by_name("TestDataset", file_index_list=None)
        iter = testSet.__iter__()
        (t1, r1, m1) = next(iter)
        (t2, r2, m2) = next(iter)

        img_list1.append(tf.squeeze(t1, axis=0))
        img_list1.append(tf.squeeze(r1, axis=0))
        img_list1.append(tf.squeeze(m1, axis=0))

        img_list2.append(tf.squeeze(t2, axis=0))
        img_list2.append(tf.squeeze(r2, axis=0))
        img_list2.append(tf.squeeze(m2, axis=0))

        z1 = self.gan.forward_E(t1, r1, m1)
        z2 = self.gan.forward_E(t2, r2, m1)

        fake_m1 = tf.squeeze(self.gan.forward_G(t1, r1, z2), axis=0)
        fake_m2 = tf.squeeze(self.gan.forward_G(t2, r2, z1), axis=0)

        img_list1.append(fake_m1)
        img_list2.append(fake_m2)

        img_lists.append(img_list1)
        img_lists.append(img_list2)

        ImageUtils.plot_images(2, 4, img_lists, is_save=True, epoch_index=888)


if __name__ == '__main__':
    T = ReflectionGANTest(which_epoch=20)
