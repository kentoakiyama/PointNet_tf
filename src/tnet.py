import tensorflow as tf

from src.initializer import EyeInitializer
from src.layers import NonLinear

class TNet(tf.keras.Model):
    def __init__(self, num_points: int, k: int, activation='relu', batchnormalization: bool=True):
        super(TNet, self).__init__()

        self.k = k

        initializer = EyeInitializer(self.k)

        self.nonlinear1 = NonLinear(64, activation=activation, shared=True, batchnormalization=batchnormalization)
        self.nonlinear2 = NonLinear(128, activation=activation, shared=True, batchnormalization=batchnormalization)
        self.nonlinear3 = NonLinear(1024, activation=activation, shared=True, batchnormalization=batchnormalization)
        self.nonlinear4 = NonLinear(512, activation=activation, shared=False, batchnormalization=batchnormalization)
        self.nonlinear5 = NonLinear(256, activation=activation, shared=False, batchnormalization=batchnormalization)

        # self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=1, strides=1)
        # self.conv2 = tf.keras.layers.Conv1D(filters=128, kernel_size=1, strides=1)
        # self.conv3 = tf.keras.layers.Conv1D(filters=1024, kernel_size=1, strides=1)
    
        # self.dense1 = tf.keras.layers.Dense(512)
        # self.dense2 = tf.keras.layers.Dense(256)
        self.dense = tf.keras.layers.Dense(k**2, kernel_initializer='zeros', bias_initializer=initializer)
        
        self.maxpooling1 = tf.keras.layers.MaxPooling1D(num_points)

        # self.bn1 = tf.keras.layers.BatchNormalization()
        # self.bn2 = tf.keras.layers.BatchNormalization()
        # self.bn3 = tf.keras.layers.BatchNormalization()
        # self.bn4 = tf.keras.layers.BatchNormalization()
        # self.bn5 = tf.keras.layers.BatchNormalization()

        # self.activation1 = tf.keras.layers.Activation(activation)
        # self.activation2 = tf.keras.layers.Activation(activation)
        # self.activation3 = tf.keras.layers.Activation(activation)
        # self.activation4 = tf.keras.layers.Activation(activation)
        # self.activation5 = tf.keras.layers.Activation(activation)
        
    def call(self, input):
        out = self.nonlinear1(input)
        out = self.nonlinear2(out)
        out = self.nonlinear3(out)
        # out = self.activation1(self.bn1(self.conv1(input)))
        # out = self.activation2(self.bn2(self.conv2(out)))
        # out = self.activation3(self.bn3(self.conv3(out)))
        out = self.maxpooling1(out)
        out = self.nonlinear4(out)
        out = self.nonlinear5(out)
        # out = self.activation4(self.bn4(self.dense1(out)))
        # out = self.activation5(self.bn5(self.dense2(out)))
    
        out = self.dense(out)

        out = tf.reshape(out, [-1, self.k, self.k])
        return out
