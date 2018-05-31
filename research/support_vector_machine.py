#!/usr/bin/env conda
# -*- coding: utf-8 -*-
"""
* ****************************************************************************
*      Owner: stayal0ne <elzhan.zeinulla@nu.edu.kz>                          *
*      Github: https://github.com/zelzhan                                    *
*      Created: Thu May 31 15:52:11 2018 by stayal0ne                                        *
******************************************************************************
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Importing the dataset
dataset = pd.read_csv('bank-full.csv', sep=";")

mapping1 = {"management":0, "technician" : 1, "entrepreneur":2,"admin.":3, 
           "blue-color":4, "housemaid":5, "retired":6, "self-employed":7, "services":8, 
           "student":9, "unemployed":10, "unknown": None}             #dealing with missing categorical data

mapping2 = {"divorced": 0, "married":1, "single":2, "unknown":None}

mapping3 = {'secondary' : 0,'primary' : 1, 'unknown' : None,'tertiary':2}

mapping4 = {"success":0, "failure" : 1, "unknown": None, "other":None}     

dataset['job'] = dataset['job'].map(mapping1)
dataset['marital'] = dataset['marital'].map(mapping2)
dataset['education'] = dataset['education'].map(mapping3)
dataset['poutcome'] = dataset['poutcome'].map(mapping4)

X = dataset.iloc[:, [i != 8 for i in range(16)]].values
y = dataset.iloc[:, -1].values

#fill in empty values
imp = Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
imp = imp.fit(X[:, [1, 2, 3, -1]])
X[:, [1, 2, 3, -1]] = imp.transform(X[:, [1, 2, 3, -1]])

#label encoding
label_encoder_X = LabelEncoder()
X[:, 4] = label_encoder_X.fit_transform(X[:, 4])     
X[:, 6] = label_encoder_X.fit_transform(X[:, 6])    
X[:, 7] = label_encoder_X.fit_transform(X[:, 7])         
X[:, 9] = label_encoder_X.fit_transform(X[:, 9])     
p = pd.DataFrame(X)   #fit and transform 1st column of the dataset
one_hot_encoder = OneHotEncoder(categorical_features=[1, 2, 3, 9, -1])   #create an OneHotEncoder object specifying the column
X = one_hot_encoder.fit_transform(X).toarray()              #OneHot encode
label_encoder_y = LabelEncoder()                            #same operations for the values which we want to predict
y = label_encoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
