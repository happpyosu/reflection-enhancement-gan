import numpy as np
import cv2
import os
import scipy.stats as st


class ReflectionSynthesizer:
    """
    ReflectionSynthesizer used to syn the fake training data by using a prior optical prior.
    """

    def __init__(self, t_path='../SynDataset/material/t', r_path='../SynDataset/material/r',
                 out_path='../SynDataset/out', replica=1):
        """
        constructor for ReflectionSynthesizer.
        :param t_path: transmission layer path
        :param r_path: reflection layer path
        :param out_path: output path
        :param replica: replica times, default set to 1, namely one (t, r) to one (m)
        """
        # t_path and r_path and output path
        self.t_path = t_path
        self.r_path = r_path
        self.out_path = out_path

        # check if the subdir is existed
        if not os.path.exists(os.path.join(self.out_path, 'r')):
            os.makedirs(os.path.join(self.out_path, 'r'))
        if not os.path.exists(os.path.join(self.out_path, 't')):
            os.makedirs(os.path.join(self.out_path, 't'))
        if not os.path.exists(os.path.join(self.out_path, 'rb')):
            os.makedirs(os.path.join(self.out_path, 'rb'))
        if not os.path.exists(os.path.join(self.out_path, 'm')):
            os.makedirs(os.path.join(self.out_path, 'm'))

        # replica times
        self.replica = replica

        # counter
        self.counter = 0

        # image size
        self.image_size = (256, 256)

        # list t dir and r dir
        self.t_list = os.listdir(t_path)
        self.r_list = os.listdir(r_path)

        if len(self.t_list) <= 0 or len(self.r_list) <= 0:
            raise ValueError("[ReflectionSynthesizer]: t image list or r image list should be greater than zero.")

        # blending mask
        self.g_mask = self._get_g_mask()

        self.IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]

    def _get_g_mask(self):
        g_mask = self._gkern(560, 3)
        g_mask = np.dstack((g_mask, g_mask, g_mask))
        return g_mask

    def _gkern(self, kern_len=100, nsig=1):
        """
        obtain a gaussian kernel.
        :param kern_len: kernel length
        :param nsig:
        :return:
        """
        interval = (2 * nsig + 1.) / (kern_len)
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kern_len + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel / kernel.max()
        return kernel

    def _gaussian_kernel(self, kernel_size=3, sigma=0):
        kx = cv2.getGaussianKernel(kernel_size, sigma)
        ky = cv2.getGaussianKernel(kernel_size, sigma)
        return np.multiply(kx, np.transpose(ky))

    def _two_peak_kernel(self, ker_size=3, deviation=(1, 1), sigma1=0, sigma2=0, attr=0.2):
        deviation = (3 + np.random.randint(10), 3 + np.random.randint(10))
        sigma1 = np.random.randint(3)
        sigma2 = sigma1 + np.random.randint(3)
        attr = 0.3 + 0.5 * np.random.rand()

        new_x = 2 * abs(deviation[0]) + 1 + int(ker_size / 2) * 2
        new_y = 2 * abs(deviation[1]) + 1 + int(ker_size / 2) * 2
        ker1 = np.zeros((new_x, new_y))
        ker2 = np.zeros((new_x, new_y))

        origin_x = int(new_x / 2)
        origin_y = int(new_y / 2)

        k0 = self._gaussian_kernel(ker_size, sigma=sigma1)
        k1 = attr * self._gaussian_kernel(ker_size, sigma=sigma2)

        k1_x = origin_x + deviation[0]
        k1_y = origin_y + deviation[1]

        pad = int((ker_size - 1) / 2)
        ker1[k1_x - pad:k1_x + pad + 1, k1_y - pad: k1_y + pad + 1] = k1
        ker2[origin_x - pad:origin_x + pad + 1, origin_y - pad:origin_y + pad + 1] = k0

        ker = ker1 + ker2
        return ker

    def _two_peak_blur(self, r):
        """
        blur the reflection image using the two peak kernel.
        :param r: origin reflection image
        :return:
        """
        ker = self._two_peak_kernel()
        r_blur = cv2.filter2D(r, -1, ker)
        return r_blur

    def _gaussian_blur(self, r):
        """
        blur the reflection image using the gaussian kernel.
        :param r:
        :return:
        """
        sigma = 0.1 + 5 * np.random.random()
        sz = int(2 * np.ceil(2 * sigma) + 1)
        r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
        return r_blur

    def _syn_data(self, t_origin, r_origin):
        """
        syn one image using the given t and r
        :param t: transmission layer.
        :param r: reflection layer.
        :return:
        """
        t = np.power(t_origin, 2.2)
        r = np.power(r_origin, 2.2)

        # random kind to decide which kernel to use.
        rand_kind = np.random.random()

        if rand_kind >= 0:
            r_blur = self._gaussian_blur(r)
        else:
            r_blur = self._two_peak_blur(r)

        blend = r_blur + t
        att = 1.08 + np.random.random() / 10.0

        for i in range(3):
            maski = blend[:, :, i] > 1
            mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
            r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att

        # do clipping
        r_blur[r_blur >= 1] = 1
        r_blur[r_blur <= 0] = 0

        h, w = r_blur.shape[0:2]
        neww = np.random.randint(0, 560 - w - 10)
        newh = np.random.randint(0, 560 - h - 10)
        alpha1 = self.g_mask[newh:newh + h, neww:neww + w, :]
        alpha2 = 1 - np.random.random() / 5.0
        r_blur_mask = np.multiply(r_blur, alpha1)

        blend = r_blur_mask + t * alpha2

        t = np.power(t, 1 / 2.2)
        rb = np.power(r_blur_mask, 1 / 2.2)

        blend = np.power(blend, 1 / 2.2)
        blend[blend >= 1] = 1
        blend[blend <= 0] = 0
        # r_blur(rb), blend(m)
        return rb, blend

    def __next__(self):
        rand_t = self.t_list[np.random.randint(len(self.t_list))]
        rand_r = self.r_list[np.random.randint(len(self.r_list))]

        # img_list_t = []
        # img_list_r = []
        # img_list_rb = []
        # img_list_m = []

        t_image = cv2.resize(cv2.imread(os.path.join(self.t_path, rand_t)), self.image_size,
                             interpolation=cv2.INTER_CUBIC) / 255
        r_image = cv2.resize(cv2.imread(os.path.join(self.r_path, rand_r)), self.image_size,
                             interpolation=cv2.INTER_CUBIC) / 255

        for _ in range(self.replica):
            self.counter += 1
            rb, m = self._syn_data(t_image, r_image)

            t = t_image * 255
            t.astype(np.uint8)

            r = r_image * 255
            r.astype(np.uint8)

            rb = rb * 255
            rb.astype(np.uint8)

            m = m * 255
            m.astype(np.uint8)

            cv2.imwrite(os.path.join(self.out_path, 't', str(self.counter)+'.jpg'), t)
            cv2.imwrite(os.path.join(self.out_path, 'r', str(self.counter)+'.jpg'), r)
            cv2.imwrite(os.path.join(self.out_path, 'rb', str(self.counter) + '.jpg'), rb)
            cv2.imwrite(os.path.join(self.out_path, 'm', str(self.counter) + '.jpg'), m)

        return self.counter


# test
if __name__ == '__main__':
    s = ReflectionSynthesizer(out_path='../SynDataset/out/0')
    for i in range(1000):
        print(i)
        next(s)

