import numpy as np
import math
import scipy.ndimage
from numpy.ma.core import exp
from scipy.constants.constants import pi
import cv2
from skimage.measure import structural_similarity as ssim


class MetricUtils:
    """
    This class offers common utils for computing the image metrics.
    """

    @staticmethod
    def compute_psnr(img1, img2):
        """
        Compute the Peak Signal to Noise Ratio (PSNR).
        PSNR is a metric that determines the image quality between a restored image and its ground truth.
        :param img1: image1, 4D-tensor
        :param img2: image2, 4D-tensor
        :return: np.float32
        """
        img1 = 255 * ((img1.numpy() + 1) / 2)
        img2 = 255 * ((img2.numpy() + 1) / 2)
        mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
        if mse < 1.0e-10:
            return 100
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    @staticmethod
    def compute_ssim(img1, img2):

        img1 = 255 * ((img1.numpy() + 1) / 2)
        img2 = 255 * ((img2.numpy() + 1) / 2)

        img1 = np.squeeze(img1, axis=0)
        img2 = np.squeeze(img2, axis=0)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        return ssim(img1, img2)