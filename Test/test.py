import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

x_in = np.random.random([1, 256, 256, 3])
kernel_in = np.random.random([32, 32, 3, 3])
x = tf.constant(x_in, dtype=tf.float32)
kernel = tf.constant(kernel_in, dtype=tf.float32)
res = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')

print(x.shape)
print(kernel.shape)
print(res.shape)