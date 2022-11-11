import tensorflow as tf

class NonLinear(tf.keras.Model):
    def __init__(self, num_out: int, activation: str='relu', shared: bool=True, batchnormalization: bool=True):
        '''
        num_out: number of output
        shared: if shared is True, the 1d convolution is used
        '''
        super(NonLinear, self).__init__()
        self.batchnormalization = batchnormalization
        if shared:
            self.dense = tf.keras.layers.Conv1D(filters=num_out, kernel_size=1, strides=1)
        else:
            self.dense = tf.keras.layers.Dense(num_out)
        
        if batchnormalization:
            self.bn = tf.keras.layers.BatchNormalization()

        self.activation = tf.keras.layers.Activation(activation)

    def call(self, input):
        out = self.dense(input)
        if self.batchnormalization:
            out = self.bn(out)
        return self.activation(out)
