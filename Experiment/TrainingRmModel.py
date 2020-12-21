import tensorflow as tf
import sys
sys.path.append('../')
import Network.RmModel as Rm
from utils import gpuutils


class TrainingRmModel:
    """
    The entrance of the rm model training.
    """
    def __init__(self, which_model: int):
        if which_model == 0:
            self.model = Rm.PerceptionRemovalModel()
        elif which_model == 1:
            self.model = Rm.BidirectionalRemovalModel()
        elif which_model == 2:
            self.model = Rm.MisalignedRemovalModel()

    def train(self):
        self.model.start_train_task()


if __name__ == '__main__':
    gpuutils.which_gpu_to_use(0)
    rm = TrainingRmModel(2)
    rm.train()



