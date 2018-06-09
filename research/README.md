
| Algorithms  | Best Accuracy % | Parameters
| --------------------------- | --------------- | -------------------------------------------------------------- |
|  Artificial Neural Network  |                 ||
|  Decision Tree              |     86.185      |**criterion** = 'entropy',  **train/test ratio** = 80/20        |
|  Naive Bayes                |     86.868      |**priors** = None, **train/test ratio** = 75/25                 |
|  Support Vector Machine     |     90.186      |**kernel** = 'rbf, **probability** = True                       |
|  Random Forest              |     90.170      | **trees** = 1000, **oob_score** = True, **criterion** = 'gini' |
|  Logistic Regression        |                 ||

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

| Algorithms  | Best Accuracy % | Parameters
| --------------------------- | --------------- | -------------------------------------------------------------- |
|  Artificial Neural Network  |     90.286      |**layers** = 2, **optimizer** = Adamax, **activ_fun** = relu, sigmoid, **loss** = binary_crossentropy|
|  Decision Tree              |     86.862      |**criterion** = 'entropy',  **train/test ratio** = 80/20        |
|  Naive Bayes                |     86.868      |**priors** = None, **train/test ratio** = 75/25                 |
|  Support Vector Machine     |     90.186      |**kernel** = 'rbf, **probability** = True                       |
|  Random Forest              |     90.884      | **trees** = 1000, **oob_score** = True, **criterion** = 'gini' |
|  Logistic Regression        |     86.185      ||

### Results ###
 According to the CAP curve analysis, the best result among the algorithms showed Random Forest with significant growth in accuracy and CAP curve performance

![Alt text](ROC_and_CAP_curves/Cap_graphs-1.png?raw=true "Title")


# Attempts: #
(Reasoning of choosing such parameters see in the paper)

## Naive Bayes: ##

| Attempt | Parameters                                           | Accuracy %      |
| ------- | ---------------------------------------------------- | --------------- |
|     1   |**priors** = None, **train/test ratio** = 75/25       | 87.23           |
|     2   |**priors** = None, **train/test ratio** = 80/20       | 86.79           |



## Random Forest: ##
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

| Attempt | Parameters                                                    | Accuracy %      |
| ------- | ------------------------------------------------------------- | --------------- |
|     1   |**criterion** = 'entropy', **train/test ratio** = 75/25        | 86.69           |
|     2   |**criterion** = 'entropy'  **train/test ratio** = 80/20        | 87.27           |

## Results of the other algorithms see in the source code ##
