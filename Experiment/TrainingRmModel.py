import tensorflow as tf
import sys
sys.path.append('../')
import Network.RmModel as Rm
from utils import gpuutils


class TrainingRmModel:
    def __init__(self, which_model: int):
        if which_model == 0:
            self.model = Rm.PerceptionRemovalModel()
        elif which_model == 1:
            self.model = Rm.BidirectionalRemovalModel()

    def train(self):
        self.model.start_train_task()


if __name__ == '__main__':
    gpuutils.which_gpu_to_use(0)
    rm = TrainingRmModel(1)
    rm.train()



