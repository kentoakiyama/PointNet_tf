import tensorflow as tf

from pointnet.tnet import TNet
from utils.layers import NonLinear


class PointNet(tf.keras.Model):
    def __init__(self, num_out: int, activation: str='relu', out_activation: str='softmax', batchnormalization:bool=True):
        '''
        num_out: number of output columns (if 10 class classification, num_out => 10)
        activation: ['relu', 'elu', 'selu', 'sigmoid', 'hard_sigmoid', 'softplux', 'softmax', 'tanh', 'linear']
        out_activation: ['sigmoid', 'softmax', 'linear'] if classification task, out_activation => ['sigmoid', 'softmax'], elif regression => 'linear'
        '''
        super(PointNet, self).__init__()

        self.input_tnet = TNet(k=3, activation=activation)
        self.feature_tnet = TNet(k=64, activation=activation, regularizer=True)
        
        self.nonlinear1 = NonLinear(64, activation=activation, shared=True, batchnormalization=batchnormalization)
        self.nonlinear2 = NonLinear(64, activation=activation, shared=True, batchnormalization=batchnormalization)
        self.nonlinear3 = NonLinear(64, activation=activation, shared=True, batchnormalization=batchnormalization)
        self.nonlinear4 = NonLinear(128, activation=activation, shared=True, batchnormalization=batchnormalization)
        self.nonlinear5 = NonLinear(1024, activation=activation, shared=True, batchnormalization=batchnormalization)
        self.nonlinear6 = NonLinear(512, activation=activation, shared=False, batchnormalization=batchnormalization, dropout_rate=0.3)
        self.nonlinear7 = NonLinear(256, activation=activation, shared=False, batchnormalization=batchnormalization, dropout_rate=0.3)

        self.maxpooling1 = tf.keras.layers.GlobalMaxPooling1D()

        self.dense = tf.keras.layers.Dense(num_out)
        self.activation = tf.keras.layers.Activation(out_activation)
    
    def call(self, input):
        # input transform
        matrix3 = self.input_tnet(input)
        out = tf.matmul(input, matrix3)

        out = self.nonlinear1(out)
        out = self.nonlinear2(out)
        
        # feature transform
        matrix64 = self.feature_tnet(input)
        out = tf.matmul(out, matrix64)

        out = self.nonlinear3(out)
        out = self.nonlinear4(out)
        out = self.nonlinear5(out)

        out = self.maxpooling1(out)
    
        out = self.nonlinear6(out)
        out = self.nonlinear7(out)
        return self.activation(self.dense(out))


class PointNetSeg(tf.keras.Model):
    def __init__(self, num_out: int, activation: str='relu', out_activation: str='softmax', batchnormalization:bool=True):
        '''
        num_points: number of nodes
        num_out: number of output columns (if 10 class classification, num_out => 10)
        activation: ['relu', 'elu', 'selu', 'sigmoid', 'hard_sigmoid', 'softplux', 'softmax', 'tanh', 'linear']
        out_activation: ['sigmoid', 'softmax', 'linear']
        '''
        super(PointNetSeg, self).__init__()

        self.input_tnet = TNet(k=3, activation=activation)
        self.feature_tnet = TNet(k=64, activation=activation)

        self.nonlinear1 = NonLinear(64, activation=activation, shared=True, batchnormalization=batchnormalization)
        self.nonlinear2 = NonLinear(64, activation=activation, shared=True, batchnormalization=batchnormalization)
        self.nonlinear3 = NonLinear(64, activation=activation, shared=True, batchnormalization=batchnormalization)
        self.nonlinear4 = NonLinear(128, activation=activation, shared=True, batchnormalization=batchnormalization)
        self.nonlinear5 = NonLinear(1024, activation=activation, shared=True, batchnormalization=batchnormalization)
        self.nonlinear6 = NonLinear(512, activation=activation, shared=True, batchnormalization=batchnormalization)
        self.nonlinear7 = NonLinear(256, activation=activation, shared=True, batchnormalization=batchnormalization)
        self.nonlinear8 = NonLinear(128, activation=activation, shared=True, batchnormalization=batchnormalization)
        self.nonlinear9 = NonLinear(128, activation=activation, shared=True, batchnormalization=batchnormalization)

        self.conv = tf.keras.layers.Conv1D(filters=num_out, kernel_size=1, strides=1)

        self.softmax = tf.keras.layers.Activation(out_activation)

        self.maxpooling1 = tf.keras.layers.GlobalMaxPooling1D()

    def call(self, input):
        shapes = tf.shape(input)
        B, N = shapes[0], shapes[1]

        # input transform
        matrix3 = self.input_tnet(input)
        out = tf.matmul(input, matrix3)
        
        out = self.nonlinear1(out)
        out = self.nonlinear2(out)
        
        # feature transform
        matrix64 = self.feature_tnet(input)
        f_out = tf.matmul(out, matrix64)
        
        out = self.nonlinear3(out)
        out = self.nonlinear4(out)
        out = self.nonlinear5(out)
        # global feature
        out = self.maxpooling1(out)
        
        # segmentation network
        out = tf.reshape(out, [B, 1, -1])
        out = tf.tile(out, [1, N, 1])
        out = tf.concat([out, f_out], axis=2)

        out = self.nonlinear6(out)
        out = self.nonlinear7(out)
        out = self.nonlinear8(out)

        out = self.nonlinear9(out)

        return self.softmax(self.conv(out))