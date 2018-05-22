#!/usr/bin/env conda
# -*- coding: utf-8 -*-
"""
* ****************************************************************************
*      Owner: stayal0ne <elzhan.zeinulla@nu.edu.kz>                          *
*      Github: https://github.com/zelzhan                                    *
*      Created: Tue May 22 05:06:23 2018 by stayal0ne                                        *
******************************************************************************
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# importing dataset
dataset = pd.read_csv('/home/stayal0ne/Machine-learning/datasets/banking.csv')
mapping = {"no": 0, "yes": 1, "unknown": None}             #dealing with missing categorical data
dataset['loan'] = dataset['loan'].map(mapping)
X = dataset.iloc[:, [1, 2, 6, 11]].values
y = dataset.iloc[:, -1].values

# fill in missing values
imp = Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
imp = imp.fit(X[:, [2]])
X[:, [2]] = imp.transform(X[:, [2]])

# encoding categorial data
label_encoder_X = LabelEncoder()
X[:, 1] = label_encoder_X.fit_transform(X[:, 1])       
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])             #fit and transform 1st column of the dataset
one_hot_encoder = OneHotEncoder(categorical_features=[0, 1])          #create an OneHotEncoder object specifying the column
X = one_hot_encoder.fit_transform(X).toarray()                        #OneHot encode
df = pd.DataFrame(X) 

# splitting the dataset into the training and test sets
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

# feature scaling
scale = StandardScaler()
X_train[:, [16, 17]] = scale.fit_transform(X_train[:, [16, 17]])
X_test[:, [16, 17]] = scale.transform(X_test[:, [16, 17]])

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

plt.countourf(X1, X2, classifier.predict(
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



