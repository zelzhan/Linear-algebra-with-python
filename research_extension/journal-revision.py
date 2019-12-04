# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 21:41:03 2019

@author: stayal0ne
"""

#from slope import slope_model
#from thal import thal_model
#from ca import ca_model

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
dataset = pd.read_csv("final.csv", encoding="utf-8-sig")

df = pd.DataFrame(columns=['age', 'sex', 'cp', 'trestbps', 'chol' , 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'prediction'])


df1 = dataset[(((dataset['-1'] == -1.0) & (dataset['-1.1'] != -1.0)) & (dataset['-1.2'] != -1.0) \
        | ((dataset['-1'] != -1.0) & (dataset['-1.1'] == -1.0)) & (dataset['-1.2'] != -1.0)) \
        | ((dataset['-1'] != -1.0) & (dataset['-1.1'] != -1.0)) & (dataset['-1.2'] == -1.0)]    



X = df1

for i in range(100):
    print(i)
    X = X.iloc[:, :].values


    #fill in empty values
    imp = IterativeImputer(missing_values=-1, max_iter=35, random_state=4)
    imp = imp.fit(X[:, :])
    X[:, :] = imp.transform(X[:, :])
    
    X = pd.DataFrame(X)
    X = X.round(0)
    

df1 = X.loc[X[11] > 0]    



df2 = pd.read_csv("processed.csv")


df1.columns = df2.columns

bigdata = df1.append(df2, ignore_index=True)    


bigdata.to_csv('augmented_data.csv', encoding='utf-8', index=False)



    
    