import tensorflow as tf

from pointnet2.pointnet2_utils import SetAbstraction
from utils.layers import NonLinear


class PointNetClsSSG(tf.keras.Model):
    def __init__(self, num_out: int, activation: str='relu', out_activation: str='softmax', batchnormalization: bool=True):
        super(PointNetClsSSG, self).__init__()
        
        self.sa1 = SetAbstraction(n_points=128, n_samples=128, radius=0.2, mlps=[64, 64, 128], activation=activation, group_all=False, batchnormalization=batchnormalization)
        self.sa2 = SetAbstraction(n_points=128, n_samples=128, radius=0.2, mlps=[128, 128, 256], activation=activation, group_all=False, batchnormalization=batchnormalization)
        self.sa3 = SetAbstraction(n_points=None, n_samples=None, radius=None, mlps=[256, 512, 1024], activation=activation, group_all=True, batchnormalization=batchnormalization)

        self.nonlinear1 = NonLinear(num_out=128, activation=activation, shared=True, batchnormalization=batchnormalization, dropout_rate=0.3)
        self.nonlinear2 = NonLinear(num_out=128, activation=activation, shared=True, batchnormalization=batchnormalization, dropout_rate=0.3)

        self.dense = tf.keras.layers.Dense(num_out)
        self.activation = tf.keras.layers.Activation(out_activation)

    def call(self, inputs):
        B, N, D = inputs.shape
        if D > 3:
            xyz = inputs[:, :, :3]
            points = inputs[:, :, 3:]
        else:
            xyz = inputs
            points = None
        xyz, points = self.sa1(xyz, points)
        xyz, points = self.sa2(xyz, points)
        print(xyz.shape, points.shape)
        xyz, points = self.sa3(xyz, points)

        x = self.nonlinear1(points)
        x = self.nonlinear2(x)
        return self.activation(self.dense(x))
