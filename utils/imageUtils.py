import matplotlib.pyplot as plt
from typing import List
import tensorflow as tf
import os
'''
This file provides image operation tools.
'''


class ImageUtils:
    @staticmethod
    def plot_image(img, dir: str, mode='syn', inc=0):
        if not os.path.exists('../result/' + str(dir) + '/'):
            os.makedirs('../result/' + str(dir) + '/')

        if not os.path.exists('../result/' + str(dir) + '/' + mode):
            os.makedirs('../result/' + str(dir) + '/' + mode)

        plt.figure()
        file_path = '../result/' + str(dir) + '/' + mode + '/' + str(inc) + '.jpg'
        img = tf.squeeze(img, axis=0)

        img = (img + 1) / 2

        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()

    @staticmethod
    def plot_images(img_nums: int, z_nums: int, img_lists: List, is_save=False, epoch_index=-1):
        """
        plot some images
        :param img_lists: image list, two-dimension
        :param epoch_index: epoch index
        :param is_save: wether to save the image
        :param z_nums: z nums to generate the middle images
        :param img_nums: image nums to plot.
        :return: None
        """
        inc = 1
        length = z_nums
        plt.figure()

        for i in range(img_nums):
            for j in range(length):
                plt.subplot(img_nums, length, inc)
                inc += 1
                im = (img_lists[i][j] + 1) / 2
                plt.imshow(im)
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')

        if is_save:
            plt.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig('../save/' + str(epoch_index) + '.jpg', dpi=600)

    @staticmethod
    def save_image_tensor(img_tensor, dir: str, inc=0):
        """
        Save the image tensor. The batch dimension should be one.
        :param dir: which dir to save
        :param inc: image index.
        :param img_tensor: image tensor to save
        :return: None
        """
        if not os.path.exists('../result/' + str(dir) + '/'):
            os.makedirs('../result/' + str(dir) + '/')

        file_path = '../result/' + str(dir) + '/' + str(inc) + '.jpg'
        img_tensor = tf.squeeze(img_tensor, axis=0)
        img = tf.image.encode_jpeg(img_tensor)
        with tf.io.gfile.GFile(file_path, 'wb') as f:
            f.write(img.numpy())
