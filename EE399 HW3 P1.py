# Where my code & write-up are stored:
# https://github.com/JimmiedWhatMom69420/EE399HW3

# Data used from homework one:
# (i) Fit the data to a three layer feed forward neural network

import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras

X=np.arange(0,31)
Y=np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
            40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])

# plot data
plt.figure()
plt.title('Dataset')
plt.scatter(X,Y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Feedforward Neural Network
model = keras.Sequential([
    keras.layers.Dense(units=64, activation='relu', input_shape=[1]),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train/Mess with the model
model.fit(X, Y, epochs=1000, verbose=0)

loss, mae = model.evaluate(X, Y, verbose=0)
print(f'Mean Absolute Error : {mae:.2f}')

# (ii) Using the first 20 data points as training data, fit the neural network.
# Compute the least-square error for each of these over the training points.
# Then compute the least square error of these models on the test data which are the remaining 10 data points.

# Train first  20 points
X_train = X[:20]
Y_train = Y[:20]
# test last 10 data points
X_test = X[20:]
Y_test = Y[20:]

#Train model for (ii)
model.fit(X_train, Y_train, epochs=1000, verbose = 0)

#Evaluate the model on training data
train_predictions = model.predict(X_train)
train_errors = Y_train - train_predictions.reshape(-1)
train_mse = np.mean(train_errors**2)
print(f'Training MSE (II): {train_mse:.2f}')

# Evaluate the model on test data
test_predictions = model.predict(X_test)
test_errors = Y_test - test_predictions.reshape(-1)
test_lse = np.mean(test_errors**2)
print(f'Test MSE (II): {test_lse:.2f}')

# Repeat (II) but use the first 10 and last 10 data points as training data.
# Then fit the model to the test data (which are the 10 held out middle data points).
# Compare these results to (ii).

# Split the data into training and test sets (per the usual)
X_train = np.concatenate([X[:10], X[-10:]])
Y_train = np.concatenate([Y[:10], Y[-10:]])
#10 middle data points for testing
X_test = X[10:20]
Y_test = Y[10:20]

# Train the model
model.fit(X_train, Y_train, epochs=1000, verbose=0)

# Evaluate the model on training data
train_predictions = model.predict(X_train)
train_errors = Y_train - train_predictions.reshape(-1)
train_mse = np.mean(train_errors**2)
print(f'Training MSE (III): {train_mse:.2f}')

# Evaluate the model on test data
test_predictions = model.predict(X_test)
test_errors = Y_test - test_predictions.reshape(-1)
test_mse = np.mean(test_errors**2)
print(f'Test MSE (III): {test_mse:.2f}')

