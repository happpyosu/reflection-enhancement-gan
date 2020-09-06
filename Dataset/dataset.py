import tensorflow as tf
import os


class DatasetFactory:
    """
        Dataset factory class. This class offers tf.dataset object by providing name
    """
    @staticmethod
    def get_dataset_by_name(name, mode="train"):
        if name == "RealDataset":
            return _RealDataset(mode=mode).get_tf_dataset()
        else:
            raise ValueError("Invalid dataset name, got '" + name + "', please check spelling mistakes.")


class _RealDataset:
    """
            This class creates a tf.dataset from producing image tuple data (t, r, m) during training.
            please make sure the dataset path is as follows:
            -dataset_root
                - r
                - t
                - m

            @Author: chen hao
            @Date:   2020.5.8
    """

    def __init__(self, root='../dataset_root/', batch_size=2, mode='train'):
        """

        :param root: the dataset root to the images that used to train the reflection image generator
        :param batch_size: image batch size for invoking tf.dataset.take method
        :param mode: specific
        """

        self.r_dir = root + mode + '/r/'
        self.t_dir = root + mode + '/t/'
        self.m_dir = root + mode + '/m/'
        self.batch_size = batch_size

        if set(os.listdir(self.r_dir)) != set(os.listdir(self.t_dir)):
            raise ValueError("the reflection image (R) files in the path: " + self.r_dir +
                             "is not consistent with the image files in transmission layer path" + self.t_dir)

        if set(os.listdir(self.m_dir)) != set(os.listdir(self.t_dir)):
            raise ValueError("the mixture image (M) files in the path: " + self.m_dir +
                             "is not consistent with the image files in transmission layer path" + self.t_dir)

        self.file_list = os.listdir(self.m_dir)
        # create the tf dataset for training
        self._tf_dataset = tf.data.Dataset. \
            from_tensor_slices(self.file_list). \
            map(self._map_fun, tf.data.experimental.AUTOTUNE). \
            batch(batch_size=batch_size, drop_remainder=True).shuffle(50, reshuffle_each_iteration=True)

    def __len__(self):
        return len(self.file_list)

    def _map_fun(self, x):
        t_path_tensor = self.t_dir + x
        r_path_tensor = self.r_dir + x
        m_path_tensor = self.m_dir + x

        img_t = tf.io.read_file(t_path_tensor)
        img_r = tf.io.read_file(r_path_tensor)
        img_m = tf.io.read_file(m_path_tensor)

        img_t = tf.image.decode_png(img_t)
        img_r = tf.image.decode_png(img_r)
        img_m = tf.image.decode_png(img_m)

        # normalize to [-1, 1]
        img_t = 2 * (tf.cast(tf.image.resize(img_t, [256, 256]), dtype=tf.float32) / 255) - 1
        img_r = 2 * (tf.cast(tf.image.resize(img_r, [256, 256]), dtype=tf.float32) / 255) - 1
        img_m = 2 * (tf.cast(tf.image.resize(img_m, [256, 256]), dtype=tf.float32) / 255) - 1

        return img_t, img_r, img_m

    def get_tf_dataset(self):
        return self._tf_dataset
