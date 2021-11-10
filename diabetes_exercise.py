""" Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line """

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

diabetes = load_diabetes()

# how many sameples and How many features?
print(diabetes.data.shape)

# What does feature s6 represent?
print(diabetes.DESCR)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, random_state=11
)

mymodel = LinearRegression()

# use fit to train our model
mymodel.fit(X_train, y_train)
# print out the coefficient
print(mymodel.coef_)
# print out the intercept
print(mymodel.intercept_)
# use predict to test your model
predited = mymodel.predict(X_test)
expected = y_test


# print out the coefficient
plt.plot(expected, predited, ".")

# plt.show()

x = np.linspace(0, 330, 100)
print(x)
y = x
plt.plot(x, y)
plt.show()

# print out the intercept


# create a scatterplot with regression line
