import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#importing the dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#fill in empty values
imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
imp = imp.fit(X[:, 1:3])
X[:, 1:3] = imp.tranform(X[:, 1:3])

# encoding categorial data
label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])            #fit and transform 1st column of the dataset
one_hot_encoder = OneHotEncoder(categorical_features=[0])   #create an OneHotEncoder object specifying the column
X = one_hot_encoder.fit_transform(X).toarray()              #OneHot encode
label_encoder_y = LabelEncoder()                            #same operations for the values which we want to predict
y = label_encoder_y.fit_transform(y)

#splitting the dataset into the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#feature scaling
scaling_X = StandardScaler()
X_train = scaling_X.fit_transform(X_train)
X_test = scaling_X.transform(X_test)
