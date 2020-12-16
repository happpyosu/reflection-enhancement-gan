import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras import layers


def gaussian_kernel(kernel_size=3, sigma=0):
    kx = cv2.getGaussianKernel(kernel_size, sigma)
    ky = cv2.getGaussianKernel(kernel_size, sigma)
    return np.multiply(kx, np.transpose(ky))

if __name__ == '__main__':
    a = [1, 2, 3]
    b = [4, 5]


