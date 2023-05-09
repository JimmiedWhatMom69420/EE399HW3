import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist

# Load the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape the images to a vector
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

# Scale the pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Compute the first 20 PCA modes
pca = PCA(n_components=20)
pca.fit(X_train)
pca_modes = pca.components_

# Visualize the PCA modes
fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(12, 9))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(pca_modes[i].reshape(28, 28), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Mode {i+1}")
plt.tight_layout()
plt.show()

# part 2

import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate the model on the test data
nn_pred = model.predict(X_test)
nn_acc = accuracy_score(y_test, np.argmax(nn_pred, axis=1))
print(f"Neural Network accuracy: {nn_acc}")

# Train and evaluate decision tree classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print(f"Decision Tree accuracy: {dt_acc}")

# Train and evaluate SVM classifier
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
print(f"SVM accuracy: {svm_acc}")