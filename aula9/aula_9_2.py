import numpy as np
import cv2

# Loading MNIST data
import keras
mnist = keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = np.array([255 - cv2.resize(image, (14, 14)) for image in x_train])
x_test = np.array([255 - cv2.resize(image, (14, 14)) for image in x_test])

knn = cv2.ml.KNearest_create()
knn.train(x_train, cv2.ml.ROW_SAMPLE, y_train)
ret,result,neighbours,dist = knn.findNearest(test,k=5)
# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print( accuracy )