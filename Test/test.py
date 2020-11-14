import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

x_in = np.random.random([1, 256, 256, 3])

pl = layers.AveragePooling2D(pool_size=(256, 256))

out = pl(x_in)
print(out.shape)

y = x_in * out
print(y)