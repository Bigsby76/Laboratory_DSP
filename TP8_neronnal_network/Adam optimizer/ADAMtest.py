#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#importing module
import sklearn
# Importing the required module
from sklearn.metrics import r2_score
import random

# creating custom data/ x and y values
X = np.arange(-210, 210, 3) 
y = np.arange(-200, 220, 3) + np.random.normal(0,20,140)
# shape of created data
X.shape, y.shape

# fixing the size of the plot
plt.figure( figsize = (12,6))
# plotting scattered plot of linear data
plt.scatter(X, y, label = 'Dataset')
plt.legend()
plt.show()

# Splitting training and test data
X_train = X[:110]
y_train = y[:110]
X_test = X[110:]
y_test = y[110:]
# printing the input and output shapes
len(X_train), len(X_test)

# size of the plot
plt.figure( figsize = (12,6))
# plotting training set and input data
plt.scatter(X_train, y_train, c='b', label = 'Training data')
plt.scatter(X_test, y_test, c='g', label='Test set')
plt.legend()
plt.show()

# creating model and dense layer
model = tf.keras.Sequential([tf.keras.layers.InputLayer(
    input_shape=1),
    tf.keras.layers.Dense(1)])

#optimizer = tf.keras.optimizers.SGD(),#SGD-> stochastic gradient descent
opt = tf.keras.optimizers.Adam(learning_rate=0.1)
# compiling the neural network model
model.compile( loss = tf.keras.losses.mae,
              optimizer = opt,
              metrics = ['mae'])

# tensorflow run model/train model on input data
model.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=10)


# Prediction of neural network model
preds = model.predict(X_test)

# Evaluating the model
print('R score is :', r2_score(y_test, preds))

#regression
a = float(model.layers[0].get_weights()[0])
b = float(model.layers[0].get_weights()[1])
yp = (a*X) + b
print('a is :', a)
print('b is :', b)

##Plot the result
# size of the plot
plt.figure(figsize=(12,6))
# plots training data, test set and predictions
plt.scatter(X_train, y_train, c="b", label="Train data")
plt.scatter(X_test, y_test, c="g", label="Test set")
plt.scatter(X_test, preds, c="r", label="Predictions")
plt.plot(X, yp, c="k", label="Regression")
plt.legend()
plt.show()