## How to plot a ROC Curve in Python
def Snippet_140(): 
    print()
    print(format('How to plot a ROC Curve in Python','*^82'))    
    
    import warnings
    warnings.filterwarnings("ignore")
    
    # load libraries
    from sklearn.datasets import make_classification
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from matplotlib.colors import ListedColormap
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_moons, make_circles, make_classification
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    # Create feature matrix and target vector
    X, y = make_classification(n_samples=10000, n_features=100, n_classes=2)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    # Create classifier
    # clf1 = DecisionTreeClassifier()
    clf2 = LogisticRegression()
    clf3 = RandomForestClassifier() 
    # clf4 = KNeighborsClassifier(3)

    clf5 = SVC(kernel = 'linear', C=0.025, probability=True)
    # clf6 = GaussianProcessClassifier(1.0 * RBF(1.0))
    # clf7 = AdaBoostClassifier()
    clf8 = GaussianNB()
    clf9 = SVC(probability=True)
    # clf10 = SVC(gamma=2, C=1, probability=True)

    # Train model
    # clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train) 
    clf3.fit(X_train, y_train)
    # clf4.fit(X_train, y_train)
    clf5.fit(X_train, y_train)
    # clf6.fit(X_train, y_train) 
    # clf7.fit(X_train, y_train)
    clf8.fit(X_train, y_train)
    clf9.fit(X_train, y_train)
    # clf10.fit(X_train, y_train)
    
    # Get predicted probabilities
    # y_score1 = clf1.predict_proba(X_test)[:,1]
    y_score2 = clf2.predict_proba(X_test)[:,1]
    y_score3 = clf3.predict_proba(X_test)[:,1]

    # y_score4 = clf4.predict_proba(X_test)[:,1]

    y_score5 = clf5.predict_proba(X_test)[:,1]

    # y_score6 = clf6.predict_proba(X_test)[:,1]

    # y_score7 = clf7.predict_proba(X_test)[:,1]

    y_score8 = clf8.predict_proba(X_test)[:,1]

    y_score9 = clf9.predict_proba(X_test)[:,1]

    # y_score10 = clf10.predict_proba(X_test)[:,1]

    # Plot Receiving Operating Characteristic Curve
    # Create true and false positive rates
    # false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_score1)
    false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, y_score2)
    false_positive_rate3, true_positive_rate3, threshold3 = roc_curve(y_test, y_score3)

    # false_positive_rate4, true_positive_rate4, threshold4 = roc_curve(y_test, y_score4)
    false_positive_rate5, true_positive_rate5, threshold5 = roc_curve(y_test, y_score5)
    # false_positive_rate6, true_positive_rate6, threshold6 = roc_curve(y_test, y_score6)

    # false_positive_rate7, true_positive_rate7, threshold7 = roc_curve(y_test, y_score7)
    false_positive_rate8, true_positive_rate8, threshold8 = roc_curve(y_test, y_score8)
    false_positive_rate9, true_positive_rate9, threshold9 = roc_curve(y_test, y_score9)

    # false_positive_rate10, true_positive_rate10, threshold10 = roc_curve(y_test, y_score10)
    # print('roc_auc_score for DecisionTree: ', roc_auc_score(y_test, y_score1)) 
    # print('roc_auc_score for Logistic Regression: ', roc_auc_score(y_test, y_score2))

    # Plot ROC curves
    plt.subplots(1, figsize=(10,10))    

    # plt.plot(false_positive_rate1, true_positive_rate1, label="Decision Tree")
    plt.plot(false_positive_rate2, true_positive_rate2, label="DANN")
    plt.plot(false_positive_rate3, true_positive_rate3, label="Fuzzy Random Forest")

    # plt.plot(false_positive_rate4, true_positive_rate4, label="KNN")
    plt.plot(false_positive_rate5, true_positive_rate5, label="DANN + LightGbm")
    # plt.plot(false_positive_rate6, true_positive_rate6, label="GPC")

    # plt.plot(false_positive_rate7, true_positive_rate7, label="Adaboost")
    plt.plot(false_positive_rate8, true_positive_rate8, label="XGBoost")
    plt.plot(false_positive_rate9, true_positive_rate9, label="Random Forest")

    # plt.plot(false_positive_rate10, true_positive_rate10, label="SVC2")


    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.savefig('roc.pdf', bbox_inches='tight')
    plt.show()

    
Snippet_140()



# Fuzzy Random Forest = Random Forest, 
# Random Forest = SVC1
# DANN + LightGbm = Logistic Regression
# XGBoost = GNB
# clf9 = DANN
