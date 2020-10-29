import tensorflow as tf

if __name__ == '__main__':
    z = tf.random.normal(shape=(1, 1, 1, 8))
    z = (z * 1) + 3

    z = tf.tile(z, [1, 256, 256, 1])

    print(z)
