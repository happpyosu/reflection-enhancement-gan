import numpy as np
import math
import cv2
from scipy.signal import convolve2d
import tensorflow as tf

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
    def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
        def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
            """
            2D gaussian mask - should give the same result as MATLAB's
            fspecial('gaussian',[shape],[sigma])
            """
            m, n = [(ss - 1.) / 2. for ss in shape]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            return h

        def filter2(x, kernel, mode='same'):
            return convolve2d(x, np.rot90(kernel, 2), mode=mode)

        im1 = tf.squeeze(im1, axis=0)
        im2 = tf.squeeze(im2, axis=0)

        im1 = 255 * ((im1.numpy() + 1) / 2)
        im2 = 255 * ((im2.numpy() + 1) / 2)

        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        if not im1.shape == im2.shape:
            raise ValueError("Input Imagees must have the same dimensions")
        if len(im1.shape) > 2:
            raise ValueError("Please input the images with 1 channel")

        M, N = im1.shape
        C1 = (k1 * L) ** 2
        C2 = (k2 * L) ** 2
        window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
        window = window / np.sum(np.sum(window))

        if im1.dtype == np.uint8:
            im1 = np.double(im1)
        if im2.dtype == np.uint8:
            im2 = np.double(im2)

        mu1 = filter2(im1, window, 'valid')
        mu2 = filter2(im2, window, 'valid')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
        sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
        sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return np.mean(np.mean(ssim_map))
