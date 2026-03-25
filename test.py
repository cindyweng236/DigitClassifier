import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def displayImage(image_data):
    plt.imshow(image_data.reshape((28, 28)))
    plt.show()

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = tf.keras.models.load_model('digit.h5')

displayImage(x_test[0])
prediction = model.predict(np.array([x_test[0]], dtype = float))

print("prediction : {}, label : {}".format(prediction, y_test[0]) )