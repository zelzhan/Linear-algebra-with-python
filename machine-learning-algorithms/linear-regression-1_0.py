# from sklearn.linear_model import LinearRegression as lm
from sklearn import datasets
from sklearn import linear_model
import pandas as pd

data = datasets.load_boston()

df = pd.DataFrame(data = data.data, columns=data.feature_names)
target = pd.DataFrame(data = data.target, columns=["Median"])

# print(df)

lm = linear_model.LinearRegression()

x = df
y = target["Median"]

model = lm.fit(x, y)

prediction = lm.predict(x)
print(prediction)
print(lm.score(x, y))
print(lm.coef_)
print(lm.intercept_)

# print(target)
# print(df)
# print(df)
#df = pd.DataFrame(data.data, columns=data.feature_names)
