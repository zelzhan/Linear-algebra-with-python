
| Algorithms before boosting  | Best Accuracy % | Parameters
| --------------------------- | --------------- | -------------------------------------------------------------- |
|  Artificial Neural Network  |                 ||
|  Decision Tree              |                 ||
|  Naive Bayes                |                 ||
|  Support Vector Machine     |     86.03       ||
|  Random Forest              |     95.75       | **trees** = 1000, **oob_score** = True, **criterion** = 'gini' |

# Attempts: #
(Reasoning of choosing such parameters see in the paper)

## Naive Bayes: ##

| Attempt | Parameters                  | Accuracy %      | |
| ------- | --------------------------- | --------------- | -------------------------------------------------------------- |
|     1   |**priors** = None          | 86.03           ||
|     2   |                 ||



## Random Forest: ##

| Attempt | Parameters                                                    | Accuracy %      | |
| ------- | ------------------------------------------------------------- | --------------- | -------------------------------------------------------------- |
|     1   |**trees** = 1000, **oob_score** = True, **criterion** = 'gini' | 95.75           ||
|     2   |                 ||

