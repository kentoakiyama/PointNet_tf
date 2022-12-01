import math

import tensorflow as tf

from utils.initializer import EyeInitializer
from utils.layers import NonLinear

class TNet(tf.keras.Model):
    def __init__(self, k: int, activation='relu', batchnormalization: bool=True, regularizer: bool=False):
        super(TNet, self).__init__()

        self.k = k

        initializer = EyeInitializer(self.k)

        self.nonlinear1 = NonLinear(64, activation=activation, shared=True, batchnormalization=batchnormalization)
        self.nonlinear2 = NonLinear(128, activation=activation, shared=True, batchnormalization=batchnormalization)
        self.nonlinear3 = NonLinear(1024, activation=activation, shared=True, batchnormalization=batchnormalization)
        self.nonlinear4 = NonLinear(512, activation=activation, shared=False, batchnormalization=batchnormalization)
        self.nonlinear5 = NonLinear(256, activation=activation, shared=False, batchnormalization=batchnormalization)

        self.maxpooling1 = tf.keras.layers.GlobalMaxPooling1D()
        
        if regularizer:
            orthogonal_reguralizer = OrthogonalReguralizer(k, reg_weight=0.001)
            self.dense = tf.keras.layers.Dense(k**2, kernel_initializer='zeros', bias_initializer=initializer, activity_regularizer=orthogonal_reguralizer)
        else:
            self.dense = tf.keras.layers.Dense(k**2, kernel_initializer='zeros', bias_initializer=initializer)

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

@tf.keras.utils.register_keras_serializable(package='Custom', name='orthogonalreguralizer2')
class OrthogonalReguralizer(tf.keras.regularizers.Regularizer):
    def __init__(self, n_features: int, reg_weight:float=0.001):
        self.n_features = n_features
        self.reg_weight = reg_weight
        self.eye = tf.eye(self.n_features)

    def __call__(self, x):
        x = tf.reshape(x, [-1, self.n_features, self.n_features])
        xt = tf.transpose(x, perm=[0, 2, 1])
        xxt = tf.matmul(x, xt)
        return tf.reduce_sum(self.reg_weight * tf.square(xxt - self.eye))

    def get_config(self):
        return {'n_features': int(self.n_features), 'reg_weight': float(self.reg_weight)}