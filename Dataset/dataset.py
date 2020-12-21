import tensorflow as tf
import os
import numpy as np

class DatasetFactory:
    """
        Dataset factory class. This class offers tf.dataset object by providing name
    """
    @staticmethod
    def get_dataset_by_name(name, mode="train", batch_size=2, file_index_list=None):
        if name == "RealDataset":
            return _RealDataset(mode=mode, batch_size=batch_size).get_tf_dataset()
        elif name == 'SynDataset':
            return _SynDataset(mode=mode, batch_size=1).get_tf_dataset()
        elif name == 'TestDataset':
            return _TestDataset(batch_size=1, file_index_list=file_index_list).get_tf_dataset()
        else:
            raise ValueError("Invalid dataset name, got '" + name + "', please check spelling mistakes.")

class _TestDataset:
    """
    This class used to offer data pipline for model testing and ablation study.
    """
    def __init__(self, root='../dataset-root/', batch_size=1, file_index_list=None):
        """

        :param root: the dataset root to the images that used to train the reflection image generator
        :param batch_size: image batch size for invoking tf.dataset.take method
        :param mode: specific
        """

        self.r_dir = root + 'test' + '/r/'
        self.rb_dir = root + 'test' + '/rb/'
        self.t_dir = root + 'test' + '/t/'
        self.m_dir = root + 'test' + '/m/'
        self.batch_size = batch_size

        if set(os.listdir(self.r_dir)) != set(os.listdir(self.t_dir)):
            raise ValueError("the reflection image (R) files in the path: " + self.r_dir +
                             "is not consistent with the image files in transmission layer path" + self.t_dir)

        if set(os.listdir(self.m_dir)) != set(os.listdir(self.t_dir)):
            raise ValueError("the mixture image (M) files in the path: " + self.m_dir +
                             "is not consistent with the image files in transmission layer path" + self.t_dir)

        # if the file index list is None, default to test all the images in the test dir.
        if file_index_list is None:
            self.file_list = os.listdir(self.m_dir)
        else:
            self.file_list = [str(x) + '.jpg' for x in file_index_list]

        # create the tf dataset for training
        self._tf_dataset = tf.data.Dataset. \
            from_tensor_slices(self.file_list). \
            map(self._map_fun, tf.data.experimental.AUTOTUNE). \
            batch(batch_size=self.batch_size, drop_remainder=True).shuffle(50, reshuffle_each_iteration=True)

    def _map_fun(self, x):
        t_path_tensor = self.t_dir + x
        r_path_tensor = self.r_dir + x
        m_path_tensor = self.m_dir + x

        img_t = tf.io.read_file(t_path_tensor)
        img_r = tf.io.read_file(r_path_tensor)
        img_m = tf.io.read_file(m_path_tensor)

        img_t = tf.image.decode_jpeg(img_t)
        img_r = tf.image.decode_jpeg(img_r)
        img_m = tf.image.decode_jpeg(img_m)

        # normalize to [-1, 1]
        img_t = 2 * (tf.cast(tf.image.resize(img_t, [256, 256]), dtype=tf.float32) / 255) - 1
        img_r = 2 * (tf.cast(tf.image.resize(img_r, [256, 256]), dtype=tf.float32) / 255) - 1
        img_m = 2 * (tf.cast(tf.image.resize(img_m, [256, 256]), dtype=tf.float32) / 255) - 1

        return img_t, img_r, img_m

    def get_tf_dataset(self):
        return self._tf_dataset


class _SynDataset:
    """
    This class creates a tf.dataset from producing image tuple data (t, r, m) during training.
    please make sure the dataset path is as follows:
    - dataset_root
        - t (transmission layer)
        - r (reflection layer)
        - rb (reflection blur layer)
        - m (mixture of t and r)
    """
    def __init__(self, root='../dataset-root/', batch_size=1, mode='train'):
        """

        :param root: the dataset root to the images that used to train the reflection image generator
        :param batch_size: image batch size for invoking tf.dataset.take method
        :param mode: specific
        """

        self.r_dir = root + mode + '/r/'
        self.rb_dir = root + mode + '/rb/'
        self.t_dir = root + mode + '/t/'
        self.m_dir = root + mode + '/m/'
        self.batch_size = batch_size

        # set batch-size to 1 if the model is in val mode.
        if mode == 'val':
            self.batch_size = 1

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
            batch(batch_size=self.batch_size, drop_remainder=True).shuffle(50, reshuffle_each_iteration=True)

    def __len__(self):
        return len(self.file_list)

    def _map_fun(self, x):
        t_path_tensor = self.t_dir + x
        r_path_tensor = self.r_dir + x
        rb_path_tensor = self.rb_dir + x
        m_path_tensor = self.m_dir + x

        img_t = tf.io.read_file(t_path_tensor)
        img_r = tf.io.read_file(r_path_tensor)
        img_rb = tf.io.read_file(rb_path_tensor)
        img_m = tf.io.read_file(m_path_tensor)

        img_t = tf.image.decode_jpeg(img_t)
        img_r = tf.image.decode_jpeg(img_r)
        img_rb = tf.image.decode_jpeg(img_rb)
        img_m = tf.image.decode_jpeg(img_m)

        # normalize to [-1, 1]
        img_t = 2 * (tf.cast(tf.image.resize(img_t, [256, 256]), dtype=tf.float32) / 255) - 1
        img_r = 2 * (tf.cast(tf.image.resize(img_r, [256, 256]), dtype=tf.float32) / 255) - 1
        img_rb = 2 * (tf.cast(tf.image.resize(img_rb, [256, 256]), dtype=tf.float32) / 255) - 1
        img_m = 2 * (tf.cast(tf.image.resize(img_m, [256, 256]), dtype=tf.float32) / 255) - 1

        return img_t, img_r, img_m

    def get_tf_dataset(self):
        return self._tf_dataset


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

    def __init__(self, root='../dataset-root/', batch_size=2, mode='train'):
        """

        :param root: the dataset root to the images that used to train the reflection image generator
        :param batch_size: image batch size for invoking tf.dataset.take method
        :param mode: specific
        """

        self.r_dir = root + mode + '/r/'
        self.t_dir = root + mode + '/t/'
        self.m_dir = root + mode + '/m/'
        self.batch_size = batch_size

        # set batch-size to 1 if the model is in val mode.
        if mode == 'val':
            self.batch_size = 1

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
            batch(batch_size=self.batch_size, drop_remainder=True).shuffle(50, reshuffle_each_iteration=True)

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


class CategoricalReflectionDataset:
    """
    This class used to build Categorical reflection image dataset.
    The dataset should be placed using the following file tree structure.
    n refer to the nums of reflection type in the dataset.
    - train
        - 1
            - t
            - r
            - m

        - 2
            - t
            - r
            - m
        - 3
            - t
            - r
            - m
        ...
        - n
    This class can produce
    """
    def __init__(self, root='../dataset-root/train_info', step_per_epoch=4000):
        # batch size
        self.batch_size = 1

        # dataset root
        self.root = root

        # total step for each epoch
        self.step_per_epoch = step_per_epoch

        dirs = os.listdir(root)

        # reflection classes nums
        self.classes_num = len(dirs)

        # used to record how many images are in each sub-dataset
        self.file_lists = []
        self.len_list = []

        # init file list
        for dir in dirs:
            path = os.path.join(root, dir, 'm')
            files = os.listdir(path)
            self.file_lists.append(files)
            self.len_list.append(len(files))

        # inc counter
        self.inc = 0

    def __iter__(self):
        self.inc = 0
        return self

    def __next__(self):
        self.inc += 1

        if self.inc > self.step_per_epoch:
            raise StopIteration

        # randomly select a class
        which_class = np.random.randint(0, self.classes_num)

        # randomly select a image
        which_image = np.random.randint(0, self.len_list[which_class])

        # image file name
        file_name = self.file_lists[which_class][which_image]

        t_path_tensor = os.path.join(self.root, str(which_class), 't', file_name)
        r_path_tensor = os.path.join(self.root, str(which_class), 'r', file_name)
        m_path_tensor = os.path.join(self.root, str(which_class), 'm', file_name)

        img_t = tf.io.read_file(t_path_tensor)
        img_r = tf.io.read_file(r_path_tensor)
        img_m = tf.io.read_file(m_path_tensor)

        img_t = tf.image.decode_jpeg(img_t)
        img_r = tf.image.decode_jpeg(img_r)
        img_m = tf.image.decode_jpeg(img_m)

        # normalize to [-1, 1]
        img_t = 2 * (tf.cast(tf.image.resize(img_t, [256, 256]), dtype=tf.float32) / 255) - 1
        img_r = 2 * (tf.cast(tf.image.resize(img_r, [256, 256]), dtype=tf.float32) / 255) - 1
        img_m = 2 * (tf.cast(tf.image.resize(img_m, [256, 256]), dtype=tf.float32) / 255) - 1

        img_t = tf.expand_dims(img_t, axis=0)
        img_r = tf.expand_dims(img_r, axis=0)
        img_m = tf.expand_dims(img_m, axis=0)

        return img_t, img_r, img_m, tf.one_hot([which_class], self.classes_num)










