import numpy as np
import math
import scipy.ndimage
from numpy.ma.core import exp
from scipy.constants.constants import pi


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

        # Variables for Gaussian kernel definition
        gaussian_kernel_sigma = 1.5
        gaussian_kernel_width = 11
        gaussian_kernel = np.zeros((gaussian_kernel_width, gaussian_kernel_width))

        # Fill Gaussian kernel
        for i in range(gaussian_kernel_width):
            for j in range(gaussian_kernel_width):
                gaussian_kernel[i, j] = \
                    (1 / (2 * pi * (gaussian_kernel_sigma ** 2))) * \
                    exp(-(((i - 5) ** 2) + ((j - 5) ** 2)) / (2 * (gaussian_kernel_sigma ** 2)))

            # Convert image matrices to double precision (like in the Matlab version)
            img_mat_1 = img1.astype(np.float)
            img_mat_2 = img2.astype(np.float)

            # Squares of input matrices
            img_mat_1_sq = img_mat_1 ** 2
            img_mat_2_sq = img_mat_2 ** 2
            img_mat_12 = img_mat_1 * img_mat_2

            # Means obtained by Gaussian filtering of inputs
            img_mat_mu_1 = scipy.ndimage.filters.convolve(img_mat_1, gaussian_kernel)
            img_mat_mu_2 = scipy.ndimage.filters.convolve(img_mat_2, gaussian_kernel)

            # Squares of means
            img_mat_mu_1_sq = img_mat_mu_1 ** 2
            img_mat_mu_2_sq = img_mat_mu_2 ** 2
            img_mat_mu_12 = img_mat_mu_1 * img_mat_mu_2

            # Variances obtained by Gaussian filtering of inputs' squares
            img_mat_sigma_1_sq = scipy.ndimage.filters.convolve(img_mat_1_sq, gaussian_kernel)
            img_mat_sigma_2_sq = scipy.ndimage.filters.convolve(img_mat_2_sq, gaussian_kernel)

            # Covariance
            img_mat_sigma_12 = scipy.ndimage.filters.convolve(img_mat_12, gaussian_kernel)

            # Centered squares of variances
            img_mat_sigma_1_sq = img_mat_sigma_1_sq - img_mat_mu_1_sq
            img_mat_sigma_2_sq = img_mat_sigma_2_sq - img_mat_mu_2_sq
            img_mat_sigma_12 = img_mat_sigma_12 - img_mat_mu_12

            # c1/c2 constants
            # First use: manual fitting
            c_1 = 6.5025
            c_2 = 58.5225

            # Second use: change k1,k2 & c1,c2 depend on L (width of color map)
            l = 255
            k_1 = 0.01
            c_1 = (k_1 * l) ** 2
            k_2 = 0.03
            c_2 = (k_2 * l) ** 2

            # Numerator of SSIM
            num_ssim = (2 * img_mat_mu_12 + c_1) * (2 * img_mat_sigma_12 + c_2)
            # Denominator of SSIM
            den_ssim = (img_mat_mu_1_sq + img_mat_mu_2_sq + c_1) * \
                       (img_mat_sigma_1_sq + img_mat_sigma_2_sq + c_2)
            # SSIM
            ssim_map = num_ssim / den_ssim
            index = np.average(ssim_map)
            return index