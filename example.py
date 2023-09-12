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
X_train = X_train[:10000, :, :, :].astype('float32') / 255
X_test = X_test[:2000, :, :, :].astype('float32') / 255
y_train = y_train[:10000, :].astype('int64').reshape(-1,)
y_test = y_test[:2000, :].astype('int64').reshape(-1,)


# X_test_blur_2 = np.load('sample_data/X_test_blur_2.npy') # np.array([motion_blur(x, size=2) for x in X_test])
X_test_blur_5 = np.load('sample_data/X_test_blur_5.npy') # np.array([motion_blur(x, size=5) for x in X_test])
# X_test_blur_15 = np.load('sample_data/X_test_blur_15.npy') #np.array([motion_blur(x, size=15) for x in X_test])


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
# embeddings_2 = ivis.transform(X_test_blur_2)
embeddings_5 = ivis.transform(X_test_blur_5)
# embeddings_15 = ivis.transform(X_test_blur_15)

# p_2 = [None, None]
# p_5 = [None, None]
# p_15 = [None, None]

# for i in range(2):
    # ks, p_2[i] = ks_2samp(embeddings_original[:, i], embeddings_2[:, i],
    #                      alternative='two-sided', mode='asymp')
    # ks, p_5[i] = ks_2samp(embeddings_original[:, i], embeddings_5[:, i],
    #                      alternative='two-sided', mode='asymp')
ks, p_5 = ks_2samp(embeddings_original[:, 1], embeddings_5[:, 1],
                         alternative='two-sided', mode='asymp')

print(f"\n\n\nScore: {ks}\n\n\n")
