import tensorflow as tf

class EyeInitializer(tf.keras.initializers.Initializer):

    def __init__(self, k):
        self.k = k

    def __call__(self, shape, dtype=None, **kwargs):
        return tf.reshape(tf.eye(self.k, dtype=dtype), [-1])
