import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
os.environ["PYTHONHASHSEED"]="1234"

import random
import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

np.random.seed(1234)
random.seed(1234)
tf.random.set_seed(1234)

from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors

from scipy.stats import ks_2samp

from ivis import Ivis


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train[:1000, :, :, :].astype('float32') / 255
X_test = X_test[:2000, :, :, :].astype('float32') / 255
y_train = y_train[:1000, :].astype('int64').reshape(-1,)
y_test = y_test[:2000, :].astype('int64').reshape(-1,)


X_test_blur_2 = np.load('sample_data/X_test_blur_2.npy') # np.array([motion_blur(x, size=2) for x in X_test])
X_test_blur_5 = np.load('sample_data/X_test_blur_5.npy') # np.array([motion_blur(x, size=5) for x in X_test])
X_test_blur_15 = np.load('sample_data/X_test_blur_15.npy') #np.array([motion_blur(x, size=15) for x in X_test])

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10))


# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# model.fit(X_train, y_train, epochs=10,
#           validation_data=(X_test, y_test))


# yh = np.argmax(model.predict(X_test), axis=-1)
# yh_blur_2 = np.argmax(model.predict(X_test_blur_2), axis=-1)
# yh_blur_5 = np.argmax(model.predict(X_test_blur_5), axis=-1)
# yh_blur_15 = np.argmax(model.predict(X_test_blur_15), axis=-1)


# print('Original Accuracy: ' + str(np.sum(y_test == yh) / len(y_test)))
# print('Blur (size=2) Accuracy: ' + str(np.sum(y_test == yh_blur_2) / len(y_test)))
# print('Blur (size=5) Accuracy: ' + str(np.sum(y_test == yh_blur_5) / len(y_test)))
# print('Blur (size=15) Accuracy: ' + str(np.sum(y_test == yh_blur_15) / len(y_test)))

ohe = OneHotEncoder()
nn = NearestNeighbors(n_neighbors=15)

y_train_ohe = ohe.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test_ohe = ohe.transform(y_test.reshape(-1, 1)).toarray()

nn.fit(y_train_ohe)

neighbour_matrix = nn.kneighbors(y_train_ohe, return_distance=False)

def create_model():
    model = tf.keras.Sequential(
      [
          tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
          tf.keras.layers.Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
          tf.keras.layers.Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
          tf.keras.layers.Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
          tf.keras.layers.Flatten()
      ])
    return  model


base_model = create_model()

ivis = Ivis(model=base_model, neighbour_matrix=neighbour_matrix, epochs=5)
ivis.fit(X_train)

embeddings_original = ivis.transform(X_test)
embeddings_2 = ivis.transform(X_test_blur_2)
embeddings_5 = ivis.transform(X_test_blur_5)
embeddings_15 = ivis.transform(X_test_blur_15)

p_2 = [None, None]
p_5 = [None, None]
p_15 = [None, None]

for i in range(2):
    ks, p_2[i] = ks_2samp(embeddings_original[:, i], embeddings_2[:, i],
                         alternative='two-sided', mode='asymp')
    ks, p_5[i] = ks_2samp(embeddings_original[:, i], embeddings_5[:, i],
                         alternative='two-sided', mode='asymp')
    ks, p_15[i] = ks_2samp(embeddings_original[:, i], embeddings_15[:, i],
                         alternative='two-sided', mode='asymp')


print('Blur (size=2) K-S p=' + str(p_2))
print('Blur (size=5) K-S p=' + str(p_5))
print('Blur (size=15) K-S p=' + str(p_15))
