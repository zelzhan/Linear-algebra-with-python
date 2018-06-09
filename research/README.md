
| Algorithms  | Best Accuracy % | Parameters
| --------------------------- | --------------- | -------------------------------------------------------------- |
|  Artificial Neural Network  |                 ||
|  Decision Tree              |     86.185      |**criterion** = 'entropy',  **train/test ratio** = 80/20        |
|  Naive Bayes                |     86.868      |**priors** = None, **train/test ratio** = 75/25                 |
|  Support Vector Machine     |     90.186      |**kernel** = 'rbf, **probability** = True                       |
|  Random Forest              |     90.170      | **trees** = 1000, **oob_score** = True, **criterion** = 'gini' |
|  Logistic Regression        |                 ||


![Alt text](ROC_and_CAP_curves/Cap_graphs.pdf?raw=true "Title")

# Attempts: #
(Reasoning of choosing such parameters see in the paper)

## Naive Bayes: ##

| Attempt | Parameters                                           | Accuracy %      |
| ------- | ---------------------------------------------------- | --------------- |
|     1   |**priors** = None, **train/test ratio** = 75/25       | 87.23           |
|     2   |**priors** = None, **train/test ratio** = 80/20       | 86.79           |



## Random Forest: ##
￼
￼


| Attempt | Parameters                                                       | Accuracy %      |
| ------- | ---------------------------------------------------------------- | --------------- |
|     1   |**trees** = 100, **oob_score** = True, **criterion** = 'gini'     | 89.97           |
|     2   |**trees** = 100, **oob_score** = True, **criterion** = 'entropy'  | 89.48           |
|     3   |**trees** = 500, **oob_score** = True, **criterion** = 'gini'     | 89.88           |
|     4   |**trees** = 500, **oob_score** = True, **criterion** = 'entropy'  | 89.85           |
|     5   |**trees** = 1000, **oob_score** = True, **criterion** = 'gini'    | 89.67           |
|     6   |**trees** = 1000, **oob_score** = True, **criterion** = 'entropy' | 89.55           |

## Decision Tree: ##

| Attempt | Parameters                                                    | Accuracy %      | |
| ------- | ------------------------------------------------------------- | --------------- | -------------------------------------------------------------- |
|     1   |**criterion** = 'entropy', **train/test ratio** = 75/25        | 86.69           ||
|     2   |**criterion** = 'entropy'  **train/test ratio** = 80/20        | 87.27           |


### SVM: ##
#
#| Attempt | Parameters                                                       | Accuracy %      |
#| ------- | ---------------------------------------------------------------- | --------------- |
#|     1   |**trees** = 100, **oob_score** = True, **criterion** = 'gini'     | 89.97           |
#|     2   |**trees** = 100, **oob_score** = True, **criterion** = 'entropy'  | 89.48           |
#|     3   |**trees** = 500, **oob_score** = True, **criterion** = 'gini'     | 89.88           |
#|     4   |**trees** = 500, **oob_score** = True, **criterion** = 'entropy'  | 89.85           |
#|     5   |**trees** = 1000, **oob_score** = True, **criterion** = 'gini'    | 89.67           |
#|     6   |**trees** = 1000, **oob_score** = True, **criterion** = 'entropy' | 89.55           |
