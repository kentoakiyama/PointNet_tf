import tensorflow as tf

from pointnet2.pointnet2_utils import SetAbstraction
from utils.layers import NonLinear


class PointNetClsSSG(tf.keras.Model):
    def __init__(self, num_out: int, activation: str='relu', out_activation: str='softmax', batchnormalization: bool=True):
        super(PointNetClsSSG, self).__init__()
        self.num_out = num_out
        
        self.sa1 = SetAbstraction(n_points=128, n_samples=128, radius=0.2, mlps=[64, 64, 128], activation=activation, group_all=False, batchnormalization=batchnormalization)
        self.sa2 = SetAbstraction(n_points=128, n_samples=128, radius=0.2, mlps=[128, 128, 256], activation=activation, group_all=False, batchnormalization=batchnormalization)
        self.sa3 = SetAbstraction(n_points=None, n_samples=None, radius=None, mlps=[256, 512, 1024], activation=activation, group_all=True, batchnormalization=batchnormalization)

        self.nonlinear1 = NonLinear(num_out=128, activation=activation, shared=True, batchnormalization=batchnormalization, dropout_rate=0.3)
        self.nonlinear2 = NonLinear(num_out=128, activation=activation, shared=True, batchnormalization=batchnormalization, dropout_rate=0.3)

        self.dense = tf.keras.layers.Dense(num_out)
        self.activation = tf.keras.layers.Activation(out_activation)

    def call(self, inputs):
        """
        inputs: [B, N, D]
        """
        inputs_shape = tf.shape(inputs)
        B = inputs_shape[0]

        if tf.shape(inputs)[-1] > 3:
            xyz = inputs[:, :, :3]
            points = inputs[:, :, 3:]
        else:
            xyz = inputs[:, :, :3]
            points = inputs[:, :, :3]
        xyz, points = self.sa1(xyz, points)
        xyz, points = self.sa2(xyz, points)
        xyz, points = self.sa3(xyz, points)

        x = self.nonlinear1(points)
        x = self.nonlinear2(x)
        x = self.dense(x)
        x = tf.reshape(x, [B, self.num_out])
        return self.activation(x)
