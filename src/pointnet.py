import tensorflow as tf

from src.tnet import TNet


class PointNet(tf.keras.Model):
    def __init__(self, num_points: int, num_out: int, activation: str='relu', out_activation: str='softmax'):
        '''
        num_points: number of nodes
        num_out: number of output columns (if 10 class classification, num_out => 10)
        activation: ['relu', 'elu', 'selu', 'sigmoid', 'hard_sigmoid', 'softplux', 'softmax', 'tanh', 'linear']
        out_activation: ['sigmoid', 'softmax', 'linear'] if classification task, out_activation => ['sigmoid', 'softmax'], elif regression => 'linear'
        '''
        super(PointNet, self).__init__()

        self.input_tnet = TNet(num_points=num_points, k=3, activation=activation)
        self.feature_tnet = TNet(num_points=num_points, k=64, activation=activation)

        self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=1, strides=1)
        self.conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=1, strides=1)
        self.conv3 = tf.keras.layers.Conv1D(filters=64, kernel_size=1, strides=1)
        self.conv4 = tf.keras.layers.Conv1D(filters=128, kernel_size=1, strides=1)
        self.conv5 = tf.keras.layers.Conv1D(filters=1024, kernel_size=1, strides=1)

        self.nonlinear1 = tf.keras.layers.Dense(512)
        self.nonlinear2 = tf.keras.layers.Dense(256)
        self.nonlinear3 = tf.keras.layers.Dense(num_out)

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.bn6 = tf.keras.layers.BatchNormalization()
        self.bn7 = tf.keras.layers.BatchNormalization()

        self.activation1 = tf.keras.layers.Activation(activation)
        self.activation2 = tf.keras.layers.Activation(activation)
        self.activation3 = tf.keras.layers.Activation(activation)
        self.activation4 = tf.keras.layers.Activation(activation)
        self.activation5 = tf.keras.layers.Activation(activation)
        self.activation6 = tf.keras.layers.Activation(activation)
        self.activation7 = tf.keras.layers.Activation(activation)

        self.softmax = tf.keras.layers.Activation(out_activation)

        self.maxpooling1 = tf.keras.layers.MaxPooling1D(num_points)

        self.flatten = tf.keras.layers.Flatten()

    def call(self, input):
        # input transform
        matrix3 = self.input_tnet(input)
        out = tf.matmul(input, matrix3)

        out = self.activation1(self.bn1(self.conv1(out)))
        out = self.activation2(self.bn2(self.conv2(out)))
        
        # feature transform
        matrix64 = self.feature_tnet(input)
        out = tf.matmul(out, matrix64)

        out = self.activation3(self.bn3(self.conv3(out)))
        out = self.activation4(self.bn4(self.conv4(out)))
        out = self.activation5(self.bn5(self.conv5(out)))
    
        out = self.maxpooling1(out)
        out = self.flatten(out)
    
        out = self.activation6(self.bn6(self.nonlinear1(out)))
        out = self.activation7(self.bn7(self.nonlinear2(out)))
        return self.softmax(self.nonlinear3(out))


class PointNetSeg(tf.keras.Model):
    def __init__(self, num_points: int, num_out: int, activation: str='relu', out_activation: str='softmax'):
        '''
        num_points: number of nodes
        num_out: number of output columns (if 10 class classification, num_out => 10)
        activation: ['relu', 'elu', 'selu', 'sigmoid', 'hard_sigmoid', 'softplux', 'softmax', 'tanh', 'linear']
        out_activation: ['sigmoid', 'softmax', 'linear']
        '''
        super(PointNetSeg, self).__init__()

        self.num_points = num_points

        self.input_tnet = TNet(num_points=num_points, k=3, activation=activation)
        self.feature_tnet = TNet(num_points=num_points, k=64, activation=activation)

        self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=1, strides=1)
        self.conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=1, strides=1)
        self.conv3 = tf.keras.layers.Conv1D(filters=64, kernel_size=1, strides=1)
        self.conv4 = tf.keras.layers.Conv1D(filters=128, kernel_size=1, strides=1)
        self.conv5 = tf.keras.layers.Conv1D(filters=1024, kernel_size=1, strides=1)
        self.conv6 = tf.keras.layers.Conv1D(filters=512, kernel_size=1, strides=1)
        self.conv7 = tf.keras.layers.Conv1D(filters=256, kernel_size=1, strides=1)
        self.conv8 = tf.keras.layers.Conv1D(filters=128, kernel_size=1, strides=1)
        self.conv9 = tf.keras.layers.Conv1D(filters=128, kernel_size=1, strides=1)
        self.conv10 = tf.keras.layers.Conv1D(filters=num_out, kernel_size=1, strides=1)

        self.nonlinear1 = tf.keras.layers.Dense(512)
        self.nonlinear2 = tf.keras.layers.Dense(256)

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.bn6 = tf.keras.layers.BatchNormalization()
        self.bn7 = tf.keras.layers.BatchNormalization()
        self.bn8 = tf.keras.layers.BatchNormalization()
        self.bn9 = tf.keras.layers.BatchNormalization()

        self.activation1 = tf.keras.layers.Activation(activation)
        self.activation2 = tf.keras.layers.Activation(activation)
        self.activation3 = tf.keras.layers.Activation(activation)
        self.activation4 = tf.keras.layers.Activation(activation)
        self.activation5 = tf.keras.layers.Activation(activation)
        self.activation6 = tf.keras.layers.Activation(activation)
        self.activation7 = tf.keras.layers.Activation(activation)
        self.activation8 = tf.keras.layers.Activation(activation)
        self.activation9 = tf.keras.layers.Activation(activation)

        self.softmax = tf.keras.layers.Activation(out_activation)

        self.maxpooling1 = tf.keras.layers.MaxPooling1D(num_points)

    def call(self, input):
        # input transform
        matrix3 = self.input_tnet(input)
        out = tf.matmul(input, matrix3)

        out = self.activation1(self.bn1(self.conv1(out)))
        out = self.activation2(self.bn2(self.conv2(out)))
        
        # feature transform
        matrix64 = self.feature_tnet(input)
        f_out = tf.matmul(out, matrix64)

        out = self.activation3(self.bn3(self.conv3(f_out)))
        out = self.activation4(self.bn4(self.conv4(out)))
        out = self.activation5(self.bn5(self.conv5(out)))
        
        # global feature
        out = self.maxpooling1(out)
        
        # segmentation network
        out = tf.tile(out, [1, self.num_points, 1])
        out = tf.concat([out, f_out], axis=2)

        out = self.activation6(self.bn6(self.conv6(out)))
        out = self.activation7(self.bn7(self.conv7(out)))
        out = self.activation8(self.bn8(self.conv8(out)))

        out = self.activation9(self.bn9(self.conv9(out)))
    
        return self.softmax(self.conv10(out))