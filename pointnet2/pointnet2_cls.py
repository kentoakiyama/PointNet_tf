import tensorflow as tf

from pointnet2.pointnet2_utils import SetAbstraction
from utils.layers import NonLinear


class PointNetClsSSG(tf.model.Model):
    def __init__(self, activation: str='relu', batchnormalization: bool=True):
        super(PointNetClsSSG, self).__init__()
        
        self.sa1 = SetAbstraction(n_points=128, n_samples=128, radius=0.2, mlps=[128, 128], activation=activation, batchnormalization=batchnormalization)
        self.sa2 = SetAbstraction(n_points=128, n_samples=128, radius=0.2, mlps=[128, 128], activation=activation, batchnormalization=batchnormalization)
        self.sa3 = SetAbstraction(n_points=None, n_samples=None, radius=None, mlps=[128, 128], activation=activation, batchnormalization=batchnormalization)

        self.nonlinear1 = NonLinear(num_out=128, activation=activation, shared=True, batchnormalization=batchnormalization)
        self.nonlinear2 = NonLinear(num_out=128, activation=activation, shared=True, batchnormalization=batchnormalization)

        pass

    def call(self, inputs):
        x = self.sa1(inputs)
        x = self.sa2(x)
        x = self.sa3(x)

        x = self.nonlinear1(x)
        x = self.nonlinear2(x)