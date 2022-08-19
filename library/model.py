from tensorflow import keras
from tensorflow.keras import layers
from library.custom_layers import ResBlock



class ResNetLike(keras.Model):
    def __init__(self, num_classes=100):
        super(ResNetLike, self).__init__()

        self.block1 = ResBlock([32, 32, 64])
        self.block2 = ResBlock([128, 128, 256])
        self.block3 = ResBlock([128, 256, 512])
        self.pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(512, activation='relu')
        self.drop_out= layers.Dropout(0.5)
        self.dense2 = layers.Dense(256, activation='relu')
        self.classifier = layers.Dense(num_classes)

    def call(self, input_tensor, training=False):
        x = self.block1(input_tensor, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.pool(x, training=training)
        x = self.dense1(x)
        x = self.drop_out(x, training=training)
        x = self.dense2(x)
        x = self.classifier(x)
        return x
    
    def summary(self):
        x = keras.Input(shape=(45, 45, 1))
        model = keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()