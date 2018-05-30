#!/usr/bin/env conda
# -*- coding: utf-8 -*-
"""
* ****************************************************************************
*      Owner: stayal0ne <elzhan.zeinulla@nu.edu.kz>                          *
*      Github: https://github.com/zelzhan                                    *
*      Created: Tue May 22 03:49:23 2018 by stayal0ne                                        *
******************************************************************************
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression

# importing dataset
dataset = pd.read_csv('/home/stayal0ne/Machine-learning/datasets/bank.csv', sep=';')
dataset['age'] = dataset['age'].astype(float)
dataset['balance'] = dataset['balance'].astype(float)
X = dataset.iloc[:, [0, 5]].values
y = dataset.iloc[:, -1].values

pd1 = pd.DataFrame(X)
pd2 = pd.DataFrame(y)



# splitting the dataset into the training and test sets
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

# encoding categorial data
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# feature scaling
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

# Fitting classifier into the training set
classifier = LogisticRegression(random_state=42)
classifier.fit(X_train, y_train)

# Prediction
y_predicted = classifier.predict(X_test)

# Checking the accuracy
con_matrix = confusion_matrix(y_test, y_predicted)


# visualisation of Training set
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                                stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(
                                        start = X_set[:, 1].min() - 1,
                                        stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(
        np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.xlim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("Classification set")
plt.xlabel("X-axis")
plt.ylabel("y-axis")
plt.legend
plt.show()

# visualisation of Test set
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                                stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(
                                        start = X_set[:, 1].min() - 1,
                                        stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(
        np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.xlim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("Classification set")
plt.xlabel("X-axis")
plt.ylabel("y-axis")
plt.legend
plt.show()



