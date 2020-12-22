import tensorflow as tf
import sys

sys.path.append('../')
from Network.network import Network
from Dataset.dataset import DatasetFactory
from Network import RmModel
from utils.metricUtils import MetricUtils


class EvaluatingRmModel:
    def __init__(self, model):

        # evaluation dataset of real data and syn dataset
        self.eval_real_dataset = DatasetFactory.get_dataset_by_name(name='RealEvalDataset')
        self.eval_syn_dataset = DatasetFactory.get_dataset_by_name(name='SynEvalDataset')

    def eval_PerceptionRemovalModel(self, weight_epoch: int, dataset_type='syn'):
        rm = RmModel.PerceptionRemovalModel()
        rm.load_weights(epoch=weight_epoch)
        if dataset_type == 'syn':
            for t, r, rb, m in self.eval_syn_dataset:
                pred_t = rm.forward(m)
                self.psnr(pred_t, t)
        elif dataset_type == 'real':
            for t, r, m in self.eval_syn_dataset:
                pred_t = rm.forward(m)
                self.psnr(pred_t, t)

    def psnr(self, pred, gt):
        psnr = MetricUtils.compute_psnr(pred, gt)
        print('[PSNRMetricProcessor]: PSNR: + ' + str(psnr))


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
