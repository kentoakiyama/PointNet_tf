import tensorflow as tf


class TNet(tf.keras.Model):
    def __init__(self, num_points: int, k: int, activation='relu'):
        super(TNet, self).__init__()

        self.k = k

        self.nonlinear1 = tf.keras.layers.Dense(64)
        self.nonlinear2 = tf.keras.layers.Dense(128)
        self.nonlinear3 = tf.keras.layers.Dense(1024)
        self.nonlinear4 = tf.keras.layers.Dense(1024)
        self.nonlinear5 = tf.keras.layers.Dense(512)
        self.nonlinear6 = tf.keras.layers.Dense(k**2)
        
        self.maxpooling1 = tf.keras.layers.MaxPooling1D(num_points)

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.bn5 = tf.keras.layers.BatchNormalization()

        self.activation1 = tf.keras.layers.Activation(activation)
        self.activation2 = tf.keras.layers.Activation(activation)
        self.activation3 = tf.keras.layers.Activation(activation)
        self.activation4 = tf.keras.layers.Activation(activation)
        self.activation5 = tf.keras.layers.Activation(activation)
        
    def call(self, input):
        out = self.activation1(self.bn1(self.nonlinear1(input)))
        out = self.activation2(self.bn2(self.nonlinear2(out)))
        out = self.activation3(self.bn3(self.nonlinear3(out)))
        out = self.maxpooling1(out)
        out = self.activation4(self.bn4(self.nonlinear4(out)))
        out = self.activation5(self.bn5(self.nonlinear5(out)))
        out = self.nonlinear6(out)
        out = tf.reshape(out, [-1, self.k, self.k])
        return out
