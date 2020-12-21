import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras import layers
import os


def gaussian_kernel(kernel_size=3, sigma=0):
    kx = cv2.getGaussianKernel(kernel_size, sigma)
    ky = cv2.getGaussianKernel(kernel_size, sigma)
    return np.multiply(kx, np.transpose(ky))

if __name__ == '__main__':
    print(tf.one_hot([1], depth=3))


