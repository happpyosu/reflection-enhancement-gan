import sys
sys.path.append('../')
from Experiment.TrainingREGAN import ReflectionGAN
from utils.imageUtils import ImageUtils
from Dataset.dataset import DatasetFactory
import tensorflow as tf
import shutil
import os


class ReflectionGANTest:
    def __init__(self, which_epoch):
        # init the reflection GAN
        self.gan = ReflectionGAN()

        # load epoch weights
        self.gan.load_weights(epoch=which_epoch)

        # check path exists
        if not os.path.exists('../dataset-root/test'):
            os.makedirs('../dataset-root/test')

        if not os.path.exists('../dataset-root/test/t'):
            os.makedirs('../dataset-root/test/t')

        if not os.path.exists('../dataset-root/test/r'):
            os.makedirs('../dataset-root/test/r')

        if not os.path.exists('../dataset-root/test/m'):
            os.makedirs('../dataset-root/test/m')

        # test set
        # self.testSet = DatasetFactory.get_dataset_by_name("TestDataset")

    def rand_generate(self, times_per_image=10):
        """
        randomly generate the reflection layer, with the transmission layer kept all zero
        :param times_per_image: generation times for each image in the test dataset
        :param r: reflection layer
        :return:
        """
        all_zero_t = tf.zeros(shape=(1, 256, 256, 3), dtype=tf.float32)
        eval_syn_dataset = DatasetFactory.get_dataset_by_name(name='SynEvalDataset')
        inc = 0

        for t, r, m in eval_syn_dataset:
            for _ in range(times_per_image):
                fake = self.gan.forward_G_with_random_noise(all_zero_t, r)
                inc += 1
                print(inc)
                ImageUtils.plot_image(fake, dir='random-gen', mode='syn', inc=inc)

    def modal_transfer(self, idx1: int, idx2: int):
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

        # copy target image to the test dir
        shutil.copyfile(src='../dataset-root/train/t/' + str(idx1) + '.jpg',
                        dst='../dataset-root/test/t/' + str(idx1) + '.jpg')

        shutil.copyfile(src='../dataset-root/train/t/' + str(idx2) + '.jpg',
                        dst='../dataset-root/test/t/' + str(idx2) + '.jpg')

        shutil.copyfile(src='../dataset-root/train/r/' + str(idx1) + '.jpg',
                        dst='../dataset-root/test/r/' + str(idx1) + '.jpg')

        shutil.copyfile(src='../dataset-root/train/r/' + str(idx2) + '.jpg',
                        dst='../dataset-root/test/r/' + str(idx2) + '.jpg')

        shutil.copyfile(src='../dataset-root/train/m/' + str(idx1) + '.jpg',
                        dst='../dataset-root/test/m/' + str(idx1) + '.jpg')

        shutil.copyfile(src='../dataset-root/train/m/' + str(idx2) + '.jpg',
                        dst='../dataset-root/test/m/' + str(idx2) + '.jpg')


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

        ImageUtils.plot_images(2, 4, img_lists, is_save=True, epoch_index=28)

        shutil.rmtree('../dataset-root/test/m')
        shutil.rmtree('../dataset-root/test/t')
        shutil.rmtree('../dataset-root/test/r')


if __name__ == '__main__':
    T = ReflectionGANTest(which_epoch=15)
    T.rand_generate()
    # T.modal_transfer(idx1=286, idx2=631)
