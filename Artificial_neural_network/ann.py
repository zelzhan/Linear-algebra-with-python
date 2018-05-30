#!/usr/bin/env conda
# -*- coding: utf-8 -*-
"""
* ****************************************************************************
*      Owner: stayal0ne <elzhan.zeinulla@nu.edu.kz>                          *
*      Github: https://github.com/zelzhan                                    *
*      Created: Sun May 27 16:47:05 2018 by stayal0ne                                        *
******************************************************************************
"""

import keras
from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#encoding categorical data
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# splitting the dataset into the training and test sets
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

# feature scaling
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

# Fitting classifier into the training set
classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim = 11))
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, nb_epoch = 100)

customer = np.array([[0, 0, 500, 1, 40, 3, 60000, 2,  1, 1, 50000]])
customer = scale.transform(customer)

# Prediction
y_predicted = classifier.predict(X_test)
y_predicted = (y_predicted>0.5)

customer_prediction = classifier.predict(customer)
customer_prediction = (customer_prediction>0.5)
# Checking the accuracy
con_matrix = confusion_matrix(y_test, y_predicted)