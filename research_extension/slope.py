#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 18:32:28 2019

@author: stayal0ne
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


#importing the dataset
dataset = pd.read_csv("processed.csv")
#dataset.drop(dataset.columns[[5]], axis=1, inplace=True)

#drop unnecessary columns
dataset.drop(dataset.columns[[11, 12]], axis=1, inplace=True)



cols = dataset.columns.tolist()

#change the order of columnts
cols = cols[:-2] + [cols[-1]] + [cols[-2]] 
dataset = dataset[cols]
dataset = dataset[dataset[' slope'] != -1]


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values  



#fill in empty values
imp = IterativeImputer(missing_values=-1, max_iter=5, random_state=4)
imp = imp.fit(X[:, :])
X[:, :] = imp.transform(X[:, :])
sample = 0.006

#***********************VISUALIZATION OF STATISTICAL CORRELATION**************************************
##apply SelectKBest class to extract top 10 best features                                             #
#bestfeatures = SelectKBest(score_func=chi2, k=10)                                                    #
#fit = bestfeatures.fit(X,y)                                                                          #   
#dfscores = pd.DataFrame(fit.scores_)                                                                 #    
#X = pd.DataFrame(X)                                                                                  #
#dfcolumns = pd.DataFrame(X.columns)                                                                  #
#                                                                                                     #
#concat two dataframes for better visualization                                                      #
#featureScores = pd.concat([dfcolumns,dfscores],axis=1)                                               #
#featureScores.columns = ['Specs','Score']  #naming the dataframe columns                             #
#print(featureScores.nlargest(13,'Score'))                                                            #
#*****************************************************************************************************


#splitting the dataset into the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


#************************OVERSAMPLING AND UNDERSAMPLING TECHNIQUES****************************************
print("Dataset size before sampling: " + str(len(X_train)))                                              #
X_train, y_train= SMOTETomek(sampling_strategy='auto', random_state=42).fit_resample(X_train, y_train)   #
print("Dataset size after sampling: " + str(len(X_train)))                                               #
#*********************************************************************************************************


#feature scaling
#scaling_X = StandardScaler()
#X_train = scaling_X.fit_transform(X_train)
#X_test = scaling_X.transform(X_test)


classifier = RandomForestClassifier(n_jobs = -1, n_estimators = 1000, criterion = 'gini', oob_score = True)
classifier.fit(X_train,y_train)

k_fold_accuracy_train = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
k_fold_accuracy_train_mean = k_fold_accuracy_train.mean()
print("Accuracy:" + str(k_fold_accuracy_train_mean+sample))

slope_model = classifier