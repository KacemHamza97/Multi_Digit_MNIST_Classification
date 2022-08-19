from tensorflow.keras import layers
from tensorflow.nn import relu


class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size=3, padding="same"):
        super(CNNBlock, self).__init__()
        
        self.conv = layers.Conv2D(out_channels, kernel_size, padding=padding)
        self.batch_norm = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = relu(self.batch_norm(x, training=training))
        return x
    
    
class ResBlock(layers.Layer):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        
        self.cnn1 = CNNBlock(channels[0])
        self.cnn2 = CNNBlock(channels[1])
        self.cnn3 = CNNBlock(channels[2])
        self.identity = layers.Conv2D(channels[1], 1 , padding="same")
        self.pooling = layers.MaxPooling2D()


    def call(self, input_tensor, training=False):
        x = self.cnn1(input_tensor, training=training)
        x = self.cnn2(x , training=training)
        x = self.cnn3(x + self.identity(input_tensor), training=training)
        x = self.pooling(x)
        return x