'''
These files contain data for video game ratings given out by the ESRB. 
Both files contain 34 features. 32 of these contribute to the ESRB's assigned rating. 
The data has already been split into a train and a test file for you to use to train and test your model. 
Your goal is to create a model that will take the 32 features for each game and predict a rating. 
Keep in mind that the final output should be the name of the target rating and not the number. 
Please refer to this file which should match your output.

Using methods covered in class:

1) load the dataset and use the KNeighborsClassifier to train and test your model

2) Display all wrong predicted and expected pairs

3) produce a csv file of the name of the game and the predicted rating

'''

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd

train_list = []
test_list = []
train = pd.read_csv('gameratings.csv')
test = pd.read_csv('test_esrb.csv')
y_train = train.iloc[:,-1]
x_train = train.iloc[:,1:-1]
y_test = test.iloc[:,-1]
x_test = test.iloc[:,1:-1]



linear_regression = LinearRegression()

linear_regression.fit(X = x_train, y =y_train)

predicted = linear_regression.predict(X = x_test)

expected = y_test


df = pd.DataFrame()

df["Expected"] = pd.Series(expected)

df["predicted"] = pd.Series(predicted)


print("Wrong predicted pairs")
wrong = [(p,e) for (p,e) in zip(predicted, expected) if int(p) != e]
print(wrong)
import csv
taget_class = { 1:"Everyone", 2:"Everyone 10+", 3:"Mature", 4:"Teen"}
with open("mypredictions.csv",'w+',encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["title","predictions"])
    for i in range(500):
        score = df.predicted[i]
        if score >= 4:
            score = 4
        if score <= 1:
            score = 1
        taget = taget_class[int(score)]
        x = [test.title[i],taget ]
        writer.writerow(x)