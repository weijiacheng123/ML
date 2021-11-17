

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd


from sklearn.neighbors import KNeighborsClassifier


import pandas as pd
from scipy.sparse import data
from sklearn.neighbors import KNeighborsClassifier
import csv
import numpy as np

gr = pd.read_csv("gameratings.csv")
tn = pd.read_csv("target_names.csv")
te = pd.read_csv("test_esrb.csv")

grp = pd.DataFrame(gr)
tep = pd.DataFrame(te)
#print(te_tconsole)

#print(tep.iloc[1:33])

X_train = grp.iloc[1:1897,1:33]
y_train = grp.iloc[1:1897,33]
X_test = tep.iloc[1:1897,1:33]
y_test = tep.iloc[1:1897,33]

print(X_train.shape)
print(y_train.shape)

knn = KNeighborsClassifier()

knn.fit(X=X_train, y=y_train)

predicted = knn.predict(X=X_test)
expected = y_test

#print(predicted[:20])
#print(expected[:20])

wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]
#print(wrong)
'''
#print(predicted)
list1 = predicted.tolist()
predict_dict = {'target':list1}
#print(predict_dict)
predict_dict_df = pd.DataFrame(predict_dict)
#print(predict_dict_df)
#print(predict_dict_df.target.values)
'''
df = pd.DataFrame(tep.iloc[:,0],predicted)
#print(predict_dict_df)

df.to_csv('MyPredictions.csv')


text = open('MyPredictions.csv',"r")

text = ''.join([i for i in text])
text = text.replace("1", "Everyone")
text = text.replace("2", "Everyone 10+")
text = text.replace("3", "Mature")
text = text.replace("4", "Teen")

x = open('MyPredictions.csv',"w")
x.writelines(text)
x.close()
'''
df.columns = ['Tile', 'prediction']
print(df.columns)
'''