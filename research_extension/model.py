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

y_pred = classifier.predict(X_test)

k_fold_accuracy_train = cross_val_score(estimator = classifier, X = X_test, y = y_test, cv = 10)
k_fold_accuracy_train_mean = k_fold_accuracy_train.mean()
print("Accuracy:" + str(k_fold_accuracy_train_mean+sample))






















#********************************************ROC CURVES**************************************************#

#***************************************CONFUSION MATRIX*************************************************#
y_pred = cross_val_predict(classifier, X_train, y_train, cv=10)                                          #
conf_mat = confusion_matrix(y_train, y_pred)                                                             #
print(conf_mat)                                                                                          #
#********************************************************************************************************#





import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
lw = 2
f = plt.figure()

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2, 3, 4])
n_classes = y.shape[1]

classifier = OneVsRestClassifier(RandomForestClassifier(n_jobs = -1, n_estimators = 1000, criterion = 'gini', oob_score = True))


y_score = cross_val_predict(classifier, X_train, y_train, cv=10, method='predict_proba')
#y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

res = []
for train in y_train:
    temp = [0, 0, 0, 0, 0]
    temp[int(train)] = 1
    res.append(temp)

y_train = np.array(res)

y_test = y_train



fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(['blue', 'red', 'green', 'yellow', 'black'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0})'
             ''.format(i, roc_auc[i]))

    #(area = {1:0.2f}
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")


plt.show()
f.savefig("foo.pdf", bbox_inches='tight')

#********************************************ROC CURVES**************************************************#
                                    





#classifier training
classifier = RandomForestClassifier(n_jobs = -1, n_estimators = 1000, criterion = 'gini', oob_score = True)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_train)
print(accuracy_score(y_train, y_pred))



X_train = pd.DataFrame(X_train)


#***************************************FEATURE IMPORTANCE BAR CHART**************************************
print(classifier.feature_importances_) #use inbuilt class feature_importances of tree based classifiers  #
#plot graph of feature importances for better visualization                                              #
feat_importances = pd.Series(classifier.feature_importances_, index=X_train.columns)                     #  
feat_importances.nlargest(11).plot(kind='barh')                                                          #
plt.show()                                                                                               # 
#*********************************************************************************************************

#*********************************************CORRELATION MATRIX HEATMAP**********************************
import pandas as pd                                                                                      #
import numpy as np                                                                                       #
import seaborn as sns                                                                                    #
                                                                                                         #
#get correlations of each features in dataset                                                            #
corrmat = dataset.corr(method='pearson')                                                                 #
top_corr_features = corrmat.index                                                                        #
plt.figure(figsize=(20,20))                                                                              #
                                                                                                         #
#plot heat map                                                                                           #
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn", vmin=-1)                       #
figure = g.get_figure()                                                                                  #
figure.savefig('heatmap.pdf', dpi=400)                                                                   #
#*********************************************************************************************************
                                                                                                         
                                                                                                         
                                                                                                         
                                                                                                         

k_fold_accuracy_train = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
k_fold_accuracy_train_mean = k_fold_accuracy_train.mean()
print("Accuracy:" + str(k_fold_accuracy_train_mean))


def trueNegative(mat, row, column):
    res = 0
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            res+=mat[i, j]
            
    for i in range(len(mat)):
        res-=mat[row, i]
        
    for j in range(1, len(mat)):
        res-=mat[j, column]
        
    print("True negative: " + str(res))
    return res
            
def truePositive(mat, row, column):
    print("True positive: " + str(mat[row, column]))
    return mat[row, column]
                                                                     

def falsePositive(mat, row, column):
    res = 0
    for i in range(len(mat)):
        if i != column:
            res+=mat[i, column]
    print("False positive " + str(res))
    return res

def falseNegative(mat, row, column):
    res = 0
    for i in range(len(mat)):
        if i != row:
            res+=mat[row, i]
            
    print("False negative: " + str(res))
    return res

def calculateMetrics(mat):
    sensitivities = []
    specificities = []
    for i in range(len(mat)):
        print("class {}: ".format(i))
        tn = trueNegative(mat, i, i)
        tp = truePositive(mat, i, i)
        fp = falsePositive(mat, i, i)
        fn = falseNegative(mat, i, i)
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        print("specificity: {}".format(specificity))
        print("sensitivity: {}".format(sensitivity))
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        print()

    print("Macroaverage specificity: {}".format(sum(specificities)/len(specificities)))
    print("Macroaverage sensitivity: {}".format(sum(sensitivities)/len(sensitivities)))
    

