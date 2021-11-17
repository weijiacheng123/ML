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

print(grp)
print(tep)

X_train = grp.iloc[:1894,1:33]
y_train = grp.iloc[:1894,33]
X_test = tep.iloc[:502,1:33]
y_test = tep.iloc[:502,33]

print(X_train.shape)
print(y_train.shape)

knn = KNeighborsClassifier()

knn.fit(X=X_train, y=y_train)

predicted = knn.predict(X=X_test)
expected = y_test

print(predicted)
print(expected)

wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]
print(wrong)


p_dict = dict(zip(tep.iloc[: ,0], predicted))
print(p_dict)
#df = pd.DataFrame.from_dict(p_dict, columns = ['title', 'prediction'])
df = pd.DataFrame(p_dict.items(), columns=['title', 'prediction'])
df_reset = df.set_index('title')
print(df_reset)


df_reset.to_csv('MyPredictions.csv')


text = open('MyPredictions.csv',"r")

text = ''.join([i for i in text])
text = text.replace("1", "Everyone")
text = text.replace("2", "Everyone 10+")
text = text.replace("3", "Mature")
text = text.replace("4", "Teen")

x = open('MyPredictions.csv',"w")
x.writelines(text)
x.close()
