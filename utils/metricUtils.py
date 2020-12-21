import numpy as np
import math


class MetricUtils:
    """
    This class offers common utils for computing the image metrics.
    """

    @staticmethod
    def compute_psnr(img1, img2):
        """
        Compute the Peak Signal to Noise Ratio (PSNR).
        PSNR is a metric that determines the image quality between a restored image and its ground truth.
        :param img1: image1
        :param img2: image2
        :return: np.float32
        """
        mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
        if mse < 1.0e-10:
            return 100
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
