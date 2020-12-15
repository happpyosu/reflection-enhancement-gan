import matplotlib.pyplot as plt
from typing import List

'''
This file provides image operation tools.
'''


class ImageUtils:
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

