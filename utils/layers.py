import tensorflow as tf

class NonLinear(tf.keras.Model):
    def __init__(self, num_out: int, activation: str='relu', shared: bool=True, batchnormalization: bool=True, dropout_rate: float=0.0):
        '''
        num_out: number of output
        shared: if shared is True, the 1d convolution is used
        '''
        super(NonLinear, self).__init__()
        self.batchnormalization = batchnormalization
        self.dropout_rate = dropout_rate

        if shared:
            self.dense = tf.keras.layers.Conv1D(filters=num_out, kernel_size=1, strides=1, padding="valid")
        else:
            self.dense = tf.keras.layers.Dense(num_out)
        
        if batchnormalization:
            self.bn = tf.keras.layers.BatchNormalization(momentum=0.)

        if dropout_rate > 0.0:
            self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.activation = tf.keras.layers.Activation(activation)

    def call(self, input):
        out = self.dense(input)
        if self.dropout_rate > 0.0:
            out = self.dropout(out)
        if self.batchnormalization:
            out = self.bn(out)
        return self.activation(out)
