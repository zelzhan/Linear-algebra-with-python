#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 21:06:57 2018

@author: karina
"""


# Importing the libraries
import numpy as np
import pandas as pd

#sklearn
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split


#graphing
import scikitplot as skplt
import matplotlib.pyplot as plt
from scipy import integrate

#deep learning
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier


# Importing the dataset
def import_dataset(dataset):
    dataset = pd.read_csv(dataset , sep=";")

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
    return X, y

def imputer(X):
    #fill in empty values
    imp = Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
    imp = imp.fit(X[:, [1, 2, 3, -1]])
    X[:, [1, 2, 3, -1]] = imp.transform(X[:, [1, 2, 3, -1]])
    return X

def encoder(X, y):
    #label encoding
    label_encoder_X = LabelEncoder()
    X[:, 4] = label_encoder_X.fit_transform(X[:, 4])
    X[:, 6] = label_encoder_X.fit_transform(X[:, 6])
    X[:, 7] = label_encoder_X.fit_transform(X[:, 7])
    X[:, 9] = label_encoder_X.fit_transform(X[:, 9])
    one_hot_encoder = OneHotEncoder(categorical_features=[1, 2, 3, 9, -1])   #create an OneHotEncoder object specifying the column
    X = one_hot_encoder.fit_transform(X).toarray()              #OneHot encode
    label_encoder_y = LabelEncoder()                            #same operations for the values which we want to predict
    y = label_encoder_y.fit_transform(y)
    return X, y

def split(X, y):
    # Splitting the dataset into the Training set and Test set
    return train_test_split(X, y, test_size = 0.25)

def scale(X_train, X_test):
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test





def a_train():
#creating first input and hidden layer with dropout
    layers = Sequential()
    layers.add(Dense(units=8, kernel_initializer ='glorot_uniform', activation='relu', input_dim=40 ))
    layers.add(Dropout(p=0.1))
    #creatinf second hidden layer
    layers.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'glorot_uniform'))
    layers.add(Dropout(p=0.1))
    #creating output layer
    layers.add(Dense(units=1,activation='sigmoid',kernel_initializer='glorot_uniform'))
    #compiling
    layers.compile(optimizer ='Adamax', metrics =['accuracy'], loss='binary_crossentropy' )
#    layers.fit(X_train,y_train,batch_size=25,epochs=300)

    return layers

def conf_matrix():
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

def preprocessing(dataset):
    X, y = import_dataset(dataset)
    X = imputer(X)
    X, y = encoder(X, y)
    X_train, X_test, y_train, y_test = split(X, y)
    X_train, X_test = scale(X_train, X_test)
    return X_train, X_test, y_train, y_test

def roccurve(y_test, y_proba):
    skplt.metrics.plot_roc(y_test, y_proba)
    plt.savefig('ANN_roc_fig.pdf')


def capcurve(y_values, y_preds_proba):
    num_pos_obs = np.sum(y_values)
    num_count = len(y_values)
    rate_pos_obs = float(num_pos_obs) / float(num_count)
    ideal = pd.DataFrame({'x':[0,rate_pos_obs,1],'y':[0,1,1]})
    xx = np.arange(num_count) / float(num_count - 1)

    y_cap = np.c_[y_values,y_preds_proba]
    y_cap_df_s = pd.DataFrame(data=y_cap)
    y_cap_df_s = y_cap_df_s.sort_values([1], ascending=True).reset_index(level = y_cap_df_s.index.names, drop=True)

    print(y_cap_df_s.head(20))

    yy = np.cumsum(y_cap_df_s[0]) / float(num_pos_obs)
    yy = np.append([0], yy[0:num_count-1]) #add the first curve point (0,0) : for xx=0 we have yy=0

    percent = 0.5
    row_index = int(np.trunc(num_count * percent))

    val_y1 = yy[row_index]
    val_y2 = yy[row_index+1]
    if val_y1 == val_y2:
        val = val_y1*1.0
    else:
        val_x1 = xx[row_index]
        val_x2 = xx[row_index+1]
        val = val_y1 + ((val_x2 - percent)/(val_x2 - val_x1))*(val_y2 - val_y1)

    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    ax.plot(ideal['x'],ideal['y'], color='grey', label='Perfect Model')
    ax.plot(xx,yy, color='red', label='User Model')
    ax.plot(xx,xx, color='blue', label='Random Model')
    ax.plot([percent, percent], [0.0, val], color='green', linestyle='--', linewidth=1)
    ax.plot([0, percent], [val, val], color='green', linestyle='--', linewidth=1, label=str(val*100)+'% of positive obs at '+str(percent*100)+'%')

    plt.xlim(0, 1.02)
    plt.ylim(0, 1.25)
    plt.title("ANN")
    plt.xlabel('% of the data')
    plt.ylabel('% of positive obs')
    plt.legend()

    plt.savefig('ANN_cap_graph.pdf')

if __name__ == '__main__':
    dataset = "bank.csv"
    X_train, X_test, y_train, y_test = preprocessing(dataset)

    classifier = KerasClassifier(build_fn = train,batch_size=32,epochs=300)

    classifier.fit(X_train, y_train)

    y_proba = classifier.predict(X_test)
    y_pred = (y_proba > 0.5)
    y_proba = classifier.predict_proba(X_test)

    #calculation of the k-fold accuracy
    k_fold_accuracy_train = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, scoring='accuracy')
    k_fold_accuracy_train = k_fold_accuracy_train.mean()

    k_fold_accuracy_test = cross_val_score(estimator = classifier, X = X_test, y = y_test, cv = 10, scoring='accuracy')
    k_fold_accuracy_test = k_fold_accuracy_test.mean()
    variance = k_fold_accuracy_test.std()


    #plotting the roccurve
    roccurve(y_test, y_proba)
    capcurve(y_test, y_proba)
