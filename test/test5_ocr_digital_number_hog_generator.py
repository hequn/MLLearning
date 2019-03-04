# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
from collections import Counter
import keras
from keras.datasets import mnist
# Load the dataset
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
digits = datasets.load_digits()
# Extract the features and labels
features = np.array(np.vstack((x_train,x_test)), 'int16')
labels = np.array(np.hstack((y_train,y_test)), 'int')

# Extract the hog features
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False, block_norm='L1')
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

print("Count of digits in dataset", Counter(labels))

# Create an linear SVM object
clf = LinearSVC()

# Perform the training
clf.fit(hog_features, labels)

# Save the classifier
joblib.dump(clf, "digits_cls.pkl", compress=3)