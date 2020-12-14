import tensorflow as tf
import sys
sys.path.append('../')
import Network.RmModel as Rm

if __name__ == '__main__':
    rm = Rm.PerceptionRemovalModel()
    rm.output_middle_result()

# class TrainingRmModel:
#     def __init__(self, which_model: int):
#         self.which = which_model

