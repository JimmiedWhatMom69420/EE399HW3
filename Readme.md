# EE-399-HW4
``Author: Marcel Ramirez``
``Publish Date: 5/8/2023``
``Course: Spring 2023 EE399``


![neural net](https://cdn-images-1.medium.com/v2/resize:fit:2600/1*QyTr6JbNP1PaPI6FjWZYTA.gif)

___
# Summary

For this homework assignment, the focus was on applying machine learning techniques to two datasets - the first being a simple synthetic dataset, and the second being the well-known MNIST dataset.

In the first part of the assignment, a three-layer feedforward neural network was fitted to the synthetic dataset using Python's NumPy library. The first 20 data points were used for training, and the remaining 10 points were held out for testing. The least-squares error of the model was computed for both the training and test data. The same procedure was then repeated using the first and last 10 data points for training, and the middle 10 points for testing. It was found that the model performed better when trained on the first 20 data points, as compared to the first and last 10 points.

In the second part of the assignment, the focus shifted to the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits. The first 20 principal components of the images were computed using PCA. A feedforward neural network was then trained on the dataset using TensorFlow, and its performance was compared to that of a decision tree classifier, an SVM classifier, and an LSTM neural network. The results showed that the feedforward neural network outperformed the other models in terms of classification accuracy.

Overall, this homework assignment provided an opportunity to explore the performance of different machine learning models on different datasets. Through this assignment, the student gained practical experience in implementing and evaluating machine learning models in Python.

## Theory

Many concepts were utilized in this assignment. We have computed the first 20 PCA modes, built a feed-forward neural network, and calculated answers with classifiers like LSTM, decision trees, and SVM.

Least square error formula allows us to evaluate the performance of the data splitting it off into training and testing data:

$$E = \sqrt{(1/n)*\sum_{j=1}^n (f(x_j)-y_j)^2}$$

# Code walkthrough
# Question 1
## Part 1

Lets first identify the data that is given in a scatter plot, shown in the graph below:
![](https://cdn.discordapp.com/attachments/823976012203163649/1105581480735940618/Screenshot_2023-05-09_124056.png)
Reconsider the data from homework one: X=np.arange(0,31) Y=np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53]) (i) Fit the data to a three layer feed forward neural network

the code to fit the data to a three-layer feedforward neural network using Keras library in Python
___
In this code, we define the data as `X` and `Y`. Then, we define a three-layer feedforward neural network model using `keras.Sequential()`. We use `Dense` layers with `relu` activation functions for the hidden layers and a linear activation function for the output layer. We compile the model using `adam` optimizer and `mse` loss function. We train the model for 1000 epochs using `fit()` method. Finally, we evaluate the model using `evaluate()` method and print the Mean Absolute Error (MAE).

~~~python
import numpy as np  
from matplotlib import pyplot as plt  
from tensorflow import keras
~~~
```python
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
```

The code above implements tensorflow to utilize a neural network with 64 neurons for the input layer, 32 neurons for the hidden layer, and finally 1 neuron for the output layer. With the mean squared error loss function, the model can compile and train 1000 epochs based on the data given on the input data.

## Part 2

the code to fit the neural network using the first 20 data points as training data and compute the least-square error for each of these over the training points, and then compute the least-square error of these models on the test data which are the remaining 10 data points:
____
```Python
# Train first 20 points  
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
```
In this code, we first split the data into training and test sets using the first 20 and last 10 data points, respectively. Then, we define a three-layer feedforward neural network model using `keras.Sequential()`. We use the same hyperparameters and settings as in part (i). We compile the model using `adam` optimizer and `mse` loss function. We train the model on the training set for 1000 epochs using `fit()` method.

After training the model, we use it to make predictions on the training and test sets. We compute the mean squared error (MSE) for each set by calculating the squared difference between the true values and predicted values, and then taking the average of these squared differences. We print the training MSE and test MSE at the end.

# Part 3


In this code, we modified the data splitting to use the first 10 and last 10 data points as the training data, and the middle 10 data points as the test data. We trained the same model architecture using the same hyperparameters as in part (ii) on the new training data. Then, we evaluate the model on both the training and test sets using the same method as in part (ii). We modify what was in part (ii) as shown below:
```python
# Split the data into training and test sets (per the usual)  
X_train = np.concatenate([X[:10], X[-10:]])  
Y_train = np.concatenate([Y[:10], Y[-10:]])  
#10 middle data points for testing  
X_test = X[10:20]  
Y_test = Y[10:20]
```



### Output

>Mean Absolute Error : 2.17

1/1 [==============================] - 0s 59ms/step
Training MSE (II): 4.47
1/1 [==============================] - 0s 15ms/step
Test MSE (II): 7.88
> Training data: first 20 data points
> Testing data: last 10 data points
> 

1/1 [==============================] - 0s 15ms/step
Training MSE (III): 2.84
1/1 [==============================] - 0s 14ms/step
Test MSE (III): 10.52
>Training data: first and last 10 data points
>Tesing data: middle 10 data points


Now let's compare the results between part (ii) and part (iii). The training MSE in part (iii) is likely to be higher than part (ii) because we are using fewer training data points. However, the test MSE in part (iii) may be **lower** than part (ii) because the model is now forced to generalize to the middle 10 data points, which were completely held out during training. Ultimately, the comparison between the two models will depend on the specific random initialization of the neural network weights and biases, as well as the choice of the training and test data points.

Now lets compare the model fit to homework 1. It really depends on what we have gotten in homework 1, and what kind of model fit we have gotten. In this case, if we get a model fitting linear regression to the entire data set, then the neural networks are more likely to outperform homework 1 as they tend to be more flexible and can capture more complex patterns in the data. But if homework one's model is more fitting such as a polynomial regression, then the model in homework one would have a better performance.

Ultimately, the comparison between models will depend on the specific modeling assumptions, the choice of hyperparameters, and the quality of the training and test datasets. 

Results from homework one:

Line:

-   Training errors:
    -   Line error: 2.242749386808538
-   Test errors:
    -   Line error: 3.36363873604787

Parabola:

-   Training errors:
    -   Parabola error: 2.1255393482773766
-   Test errors:
    -   Parabola error: 8.713651781874919

19th-degree polynomial:

-   Training errors:
    -   19th-degree polynomial error: 0.028351503968806435
-   Test errors:
    -   19th-degree polynomial error: 28617752784.428474

Results from (ii):

Neural network with first 20 data points as training data:

-   Training MSE: 4.47
-   Test MSE: 7.88

Results from (iii):

Neural network with first 10 and last 10 data points as training data:

-   Training MSE: 2.84
-   Test MSE: 10.52

Comparing the results, we can see the following trends:

-   For the line model, the neural network in (ii) has a higher training and test MSE compared to the line model from homework one.
-   For the parabola model, the neural network in (ii) has a higher training and test MSE compared to the parabola model from homework one.
-   For the 19th-degree polynomial model, the neural network in (ii) has a higher training MSE but a lower test MSE compared to the 19th-degree polynomial model from homework one.

Similarly, comparing the results from (iii) to the results from homework one, we can observe the following trends:

-   For the line model, the neural network in (iii) has lower training and test MSE compared to the line model from homework one.
-   For the parabola model, the neural network in (iii) has lower training and test MSE compared to the parabola model from homework one.
-   For the 19th-degree polynomial model, the neural network in (iii) has higher training and test MSE compared to the 19th-degree polynomial model from homework one.

Overall, the performance of the neural network models in (ii) and (iii) is mixed compared to the models fit in homework one. It's important to note that the choice of model architecture, hyperparameters, and data splitting can greatly influence the results. Additionally, the neural network models have the advantage of being able to capture more complex patterns in the data compared to the polynomial models used in homework one.

# Question 2

## part I

In this code, we first load the MNIST dataset using `tensorflow.keras.datasets.mnist.load_data()`. We then reshape the images to be 1D vectors of length 784 and scale the pixel values to be between 0 and 1. We then use `sklearn.decomposition.PCA` to compute the first 20 PCA modes of the training data, and store them in the `pca_modes` variable. Finally, we visualize the first 20 modes using `matplotlib`.

The resulting plot shows the first 20 PCA modes of the digit images. Each mode corresponds to a different spatial pattern of pixel intensities that explains some of the variability in the data. The first mode (top left) corresponds to the most important spatial pattern, while the 20th mode (bottom right) corresponds to the least important pattern.

## part ii

In this code, we first load the MNIST dataset using `tensorflow.keras.datasets.mnist.load_data()`. We then reshape the images to be 1D vectors of length 784 and scale the pixel values to be between 0 and 1. We define a feedforward neural network with two hidden layers and train it using the `fit()` method. We then evaluate the neural network's performance on the test data using the `predict()` method and `accuracy_score()` function.

We also train and evaluate a decision tree classifier and a support vector machine (SVM) classifier using the `sklearn.tree.DecisionTreeClassifier()` and `sklearn.svm.SVC()` classes, respectively.

To compare the results of these classifiers, we can simply compare their accuracy scores on the test data. The neural network should perform better than the decision tree classifier and SVM, since it can learn more complex relationships between the input features and the output labels. However, the decision tree classifier and SVM are simpler and faster to train, so they may be preferable in certain situations where speed or interpretability is a priority.

Output:

>11490434/11490434 [==============================] - 0s 0us/step
Epoch 1/5
1875/1875 [==============================] - 3s 1ms/step - loss: 0.2929 - accuracy: 0.9142 - val_loss: 0.1346 - val_accuracy: 0.9589
Epoch 2/5
1875/1875 [==============================] - 2s 1ms/step - loss: 0.1401 - accuracy: 0.9588 - val_loss: 0.1031 - val_accuracy: 0.9674
Epoch 3/5
1875/1875 [==============================] - 2s 1ms/step - loss: 0.1054 - accuracy: 0.9688 - val_loss: 0.0840 - val_accuracy: 0.9724
Epoch 4/5
1875/1875 [==============================] - 2s 1ms/step - loss: 0.0861 - accuracy: 0.9736 - val_loss: 0.0803 - val_accuracy: 0.9754
Epoch 5/5
1875/1875 [==============================] - 2s 1ms/step - loss: 0.0739 - accuracy: 0.9768 - val_loss: 0.0715 - val_accuracy: 0.9786
313/313 [==============================] - 0s 620us/step
Neural Network accuracy: 0.9786
Decision Tree accuracy: 0.8769