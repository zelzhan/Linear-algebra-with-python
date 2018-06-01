
| Algorithms  | Best Accuracy % | Parameters
| --------------------------- | --------------- | -------------------------------------------------------------- |
|  Artificial Neural Network  |                 ||
|  Decision Tree              |     87.27       ||
|  Naive Bayes                |     86.03       |**priors** = None|
|  Support Vector Machine     |                 ||
|  Random Forest              |     95.75       | **trees** = 1000, **oob_score** = True, **criterion** = 'gini' |

# Attempts: #
(Reasoning of choosing such parameters see in the paper)

## Naive Bayes: ##

| Attempt | Parameters                                           | Accuracy %      | |
| ------- | ---------------------------------------------------- | --------------- | -------------------------------------------------------------- |
|     1   |**priors** = None, **train/test ratio** = 75/25       | 87.23           ||
|     2   |**priors** = None, **train/test ratio** = 80/20       | 86.79|



## Random Forest: ##

| Attempt | Parameters                                                    | Accuracy %      | |
| ------- | ------------------------------------------------------------- | --------------- | -------------------------------------------------------------- |
|     1   |**trees** = 1000, **oob_score** = True, **criterion** = 'gini' |            ||
|     2   |**trees** = 1000, **oob_score** = True, **criterion** = 'gini' ||
|     3   |**trees** = 1000, **oob_score** = True, **criterion** = 'gini' ||
|     4   |**trees** = 1000, **oob_score** = True, **criterion** = 'gini' ||
|     5   |**trees** = 1000, **oob_score** = True, **criterion** = 'gini' ||
|     6   |**trees** = 1000, **oob_score** = True, **criterion** = 'gini' ||
|     7   |**trees** = 1000, **oob_score** = True, **criterion** = 'gini' ||

## Decision Tree: ##

| Attempt | Parameters                                                    | Accuracy %      | |
| ------- | ------------------------------------------------------------- | --------------- | -------------------------------------------------------------- |
|     1   |**criterion** = 'entropy', **train/test ratio** = 75/25        | 86.69           ||
|     2   |**criterion** = 'entropy'  **train/test ratio** = 80/20        | 87.27           |

