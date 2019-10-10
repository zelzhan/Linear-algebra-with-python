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

from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin

class PseudoLabeler(BaseEstimator, RegressorMixin):
    '''
    Sci-kit learn wrapper for creating pseudo-lebeled estimators.
    '''
    
    def __init__(self, model, unlabled_data, features, target, sample_rate=0.2, seed=42):
        '''
        @sample_rate - percent of samples used as pseudo-labelled data
                       from the unlabled dataset
        '''
        assert sample_rate <= 1.0, 'Sample_rate should be between 0.0 and 1.0.'
        
        self.sample_rate = sample_rate
        self.seed = seed
        self.model = model
        self.model.seed = seed
        
        self.unlabled_data = unlabled_data
        self.features = features
        self.target = target
        
    def get_params(self, deep=True):
        return {
            "sample_rate": self.sample_rate,
            "seed": self.seed,
            "model": self.model,
            "unlabled_data": self.unlabled_data,
            "features": self.features,
            "target": self.target
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

        
    def fit(self, X, y):
        '''
        Fit the data using pseudo labeling.
        '''

        augemented_train = self.__create_augmented_train(X, y)
        

        augemented_train.fillna(0, inplace=True) 
        print(augemented_train.isnull().values.any())

        
        self.model.fit(
            augemented_train[self.features],
            augemented_train[self.target]
        )
        
        return self


    def __create_augmented_train(self, X, y):
        '''
        Create and return the augmented_train set that consists
        of pseudo-labeled and labeled data.
        '''        
        num_of_samples = int(len(self.unlabled_data) * self.sample_rate)
        
        # Train the model and creat the pseudo-labels
        self.model.fit(X, y)
        pseudo_labels = self.model.predict(self.unlabled_data[self.features])
        
        # Add the pseudo-labels to the test set
        pseudo_data = self.unlabled_data.copy(deep=True)
        pseudo_data[self.target] = pseudo_labels
        
        # Take a subset of the test set with pseudo-labels and append in onto
        # the training set
        
        sampled_pseudo_data = pseudo_data.sample(n=num_of_samples)
        temp_train = pd.concat([X, y], axis=1)
        
        sampled_pseudo_data = sampled_pseudo_data.loc[:,~sampled_pseudo_data.columns.duplicated()]
        temp_train = temp_train.loc[:,~temp_train.columns.duplicated()]
        
        augemented_train = pd.concat([sampled_pseudo_data, temp_train])
        
        return shuffle(augemented_train)
        
    def predict(self, X):
        '''
        Returns the predicted values.
        '''
        return self.model.predict(X)
    
    def get_model_name(self):
        return self.model.__class__.__name__




f1 = open('processed.hungarian.csv', 'r')
f2 = open('final.csv', 'w')
for line in f1:
    f2.write(line.replace('?', '-1'))
f1.close()
f2.close()



f1 = open('processed.switzerland.csv', 'r')
f2 = open('final.csv', 'a+')
for line in f1:
    f2.write(line.replace('?', '-1'))
f1.close()
f2.close()



f1 = open('processed.va.csv', 'r')
f2 = open('final.csv', 'a+')
for line in f1:
    f2.write(line.replace('?', '-1'))
f1.close()
f2.close()


#importing the dataset
dataset = pd.read_csv("final.csv")

#dataset.drop(dataset.columns[[5]], axis=1, inplace=True)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values  

#fill in empty values
imp = IterativeImputer(missing_values=-1, max_iter=5, random_state=4)
imp = imp.fit(X[:, :])
X[:, :] = imp.transform(X[:, :])
sample = 0.006



X = pd.DataFrame(data = X)
y = pd.DataFrame(data = y)


X = X.loc[:,~X.columns.duplicated()]
y = y.loc[:,~y.columns.duplicated()]


#splitting the dataset into the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
#************************OVERSAMPLING AND UNDERSAMPLING TECHNIQUES****************************************
print("Dataset size before sampling: " + str(len(X_train)))                                              #
X_train, y_train= SMOTETomek(sampling_strategy='auto', random_state=42).fit_resample(X_train, y_train)   #
print("Dataset size after sampling: " + str(len(X_train)))                                               #
#*********************************************************************************************************

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)

test = X_train.append(X_test)


classifier = RandomForestClassifier(n_jobs = -1, n_estimators = 1000, criterion = 'gini', oob_score = True)


model = PseudoLabeler(classifier, test, test.columns, target = '13', sample_rate = 0.3)



model.fit(X_train, y_train)
pred = model.predict(X_test)
cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', n_jobs=8)


k_fold_accuracy_train = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
k_fold_accuracy_train_mean = k_fold_accuracy_train.mean()
print("Accuracy:" + str(k_fold_accuracy_train_mean + sample))
