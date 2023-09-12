import os
import cv2
os.environ["PYTHONHASHSEED"]="1234"

import random
import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

np.random.seed(1234)
random.seed(1234)
tf.random.set_seed(1234)


from ivis import Ivis

def motion_blur(image, size=5):

    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    
    output = cv2.filter2D(image, -1, kernel_motion_blur)
    return output

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
y_train = y_train.astype('int64').reshape(-1,)
y_test = y_test.astype('int64').reshape(-1,)

X_test_blur_2 = np.array([motion_blur(x, size=2) for x in X_test])
X_test_blur_5 = np.array([motion_blur(x, size=5) for x in X_test])
X_test_blur_15 = np.array([motion_blur(x, size=15) for x in X_test])

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10,
          validation_data=(X_test, y_test))


yh = np.argmax(model.predict(X_test), axis=-1)
yh_blur_2 = np.argmax(model.predict(X_test_blur_2), axis=-1)
yh_blur_5 = np.argmax(model.predict(X_test_blur_5), axis=-1)
yh_blur_15 = np.argmax(model.predict(X_test_blur_15), axis=-1)


print('Original Accuracy: ' + str(np.sum(y_test == yh) / len(y_test)))
print('Blur (size=2) Accuracy: ' + str(np.sum(y_test == yh_blur_2) / len(y_test)))
print('Blur (size=5) Accuracy: ' + str(np.sum(y_test == yh_blur_5) / len(y_test)))
print('Blur (size=15) Accuracy: ' + str(np.sum(y_test == yh_blur_15) / len(y_test)))
