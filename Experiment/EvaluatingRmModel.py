import tensorflow as tf
import sys

sys.path.append('../')
from Dataset.dataset import DatasetFactory
from Network import RmModel
from utils.metricUtils import MetricUtils
from utils import gpuutils
from utils.imageUtils import ImageUtils


class EvaluatingRmModel:
    def __init__(self):

        # evaluation dataset of real data and syn dataset
        self.eval_real_dataset = DatasetFactory.get_dataset_by_name(name='RealEvalDataset')
        # self.eval_syn_dataset = DatasetFactory.get_dataset_by_name(name='SynEvalDataset')

    def evalRmModel(self, weight_epoch: int, which_model=0, dataset_type='real'):
        if which_model == 0:
            rm = RmModel.PerceptionRemovalModel()
            name = 'percepRm'
        elif which_model == 1:
            rm = RmModel.BidirectionalRemovalModel()
            name = 'birdirRm'
        elif which_model == 2:
            rm = RmModel.MisalignedRemovalModel()
            name = 'misalignedRm'
        elif which_model == 3:
            rm = RmModel.EncoderDecoderRemovalModel()
            name = 'EncoderDecoderRm'
        else:
            raise NotImplementedError("EvaluatingRmModel: No Such Rm model!")

        rm.load_weights(epoch=weight_epoch)

        inc = 0
        avg_psnr = 0
        avg_ssim = 0

        if dataset_type == 'real':
            for t, r, m in self.eval_real_dataset:
                inc += 1
                pred_t = rm.forward(m)
                ImageUtils.save_image_tensor(pred_t, name, inc)
                avg_psnr += self.psnr(pred_t, t)
                avg_ssim += self.ssim(pred_t, t)
        print('[AVG PSNR]: AVG PSNR: + ' + str(avg_psnr))
        print('[AVG SSIM]: AVG SSIM: + ' + str(avg_ssim))
        # elif dataset_type == 'syn':
        #     for t, r, m in self.eval_syn_dataset:
        #         pred_t = rm.forward(m)
        #         self.psnr(pred_t, t)

    def psnr(self, pred, gt):
        psnr = MetricUtils.compute_psnr(pred, gt)
        print('[PSNR]: PSNR: + ' + str(psnr))

    def ssim(self, pred, gt):
        ssim = MetricUtils.compute_ssim(pred, gt)
        print('[SSIM]: SSIM: + ' + str(ssim))


class MetricProcessor:
    """
    base class for metric processor
    """

    def __init__(self, pred, gt):
        self.pred = pred
        self.gt = gt

    def eval(self):
        pass


class PSNRMetricProcessor(MetricProcessor):
    def __init__(self, pred, gt):
        super(PSNRMetricProcessor, self).__init__(pred, gt)

    def eval(self):
        psnr = MetricUtils.compute_psnr(self.pred, self.gt)
        print('[PSNRMetricProcessor]: PSNR: + ' + str(psnr))


class MetricProcessorHolder:
    def __init__(self):
        self.processor_list = []

    def add(self, processor: MetricProcessor):
        self.processor_list.append(processor)

    def run(self):
        for p in self.processor_list:
            p.eval()


if __name__ == '__main__':
    gpuutils.which_gpu_to_use(1)
    E = EvaluatingRmModel()
    E.evalRmModel(which_model=0, weight_epoch=99, dataset_type='real')
