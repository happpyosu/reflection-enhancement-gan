import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras import layers


def gaussian_kernel(kernel_size=3, sigma=0):
    kx = cv2.getGaussianKernel(kernel_size, sigma)
    ky = cv2.getGaussianKernel(kernel_size, sigma)
    return np.multiply(kx, np.transpose(ky))

if __name__ == '__main__':
    # tensor = tf.io.read_file('../SynDataset/out/r/21.jpg')
    # tensor = tf.image.decode_jpeg(tensor) / 255
    # sigma1 = np.random.randint(3)
    # ker = gaussian_kernel(sigma=sigma1)

    a = tf.ones(shape=(1, 256, 256, 3))
    ker = tf.ones(shape=(3, 3, 3, 2))

    op = tf.nn.conv2d(a, ker, strides=(1, 1, 1, 1), padding='SAME')

    print(op)

