import matplotlib.pyplot as plt
import numpy as np
import cv2

# Loading MNIST data
import keras
mnist = keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = np.array([255 - cv2.resize(image, (14, 14)) for image in x_train])
x_test = np.array([255 - cv2.resize(image, (14, 14)) for image in x_test])

x_train_unpacked = np.array([np.concatenate(image) for image in x_train])/255.0
x_test_unpacked = np.array([np.concatenate(image) for image in x_test])/255.0

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

n_neighbors = 5

clf = KNeighborsClassifier(n_neighbors, n_jobs=8, weights = "distance")

clf.fit(x_train_unpacked, y_train)

y_pred = clf.predict(x_test_unpacked)

accuracy = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
error = 1 - accuracy

print(f"Error rate: {error*100}%")