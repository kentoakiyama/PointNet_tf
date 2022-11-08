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

        self.dense = tf.keras.layers.Dense(k**2, kernel_initializer='zeros', bias_initializer=initializer)
        
        self.maxpooling1 = tf.keras.layers.MaxPooling1D(num_points)

    def call(self, input):
        out = self.nonlinear1(input)
        out = self.nonlinear2(out)
        out = self.nonlinear3(out)

        out = self.maxpooling1(out)
        out = self.nonlinear4(out)
        out = self.nonlinear5(out)
    
        out = self.dense(out)

        out = tf.reshape(out, [-1, self.k, self.k])
        return out
