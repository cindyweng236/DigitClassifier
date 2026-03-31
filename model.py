import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

def buildModel(activation, optimizer):
    model = tf.keras.models.Sequential([
        layers.Reshape((28, 28, 1), input_shape=(28, 28)),
        
        layers.Conv2D(32, (3, 3), activation = 'relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    

#add data augmentation
datagen = ImageDataGenerator(
    #rotate +-10°
    rotation_range=10,
    #shift horizontally +-10%
    width_shift_range=0.1,
    #shift vertically +-10%
    height_shift_range=0.1,
    #zoom +-10%
    zoom_range=0.1     
)

#add dimension to match rank 4 required for fit input
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

datagen.fit(x_train)

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(x_test, y_test)
)

model.save('digit.h5')



