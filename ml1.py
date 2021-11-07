from scipy.sparse.construct import rand
from sklearn.datasets import load_digits

digits = load_digits()

#print(digits.DESCR)
'''
print(digits.data[150])
print(digits.target[150])

print(digits.data.shape)
print(digits.target.shape)
'''

import matplotlib.pyplot as plt
figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6,4))

# plt.show()

for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)

plt.tight_layout()
#plt.show()

from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    digits.data, digits.target, random_state=11
)

print(data_train.shape)
print(data_test.shape)
print(target_train.shape)
print(target_test.shape)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X=data_train , y=target_train)

predicted = knn.predict(X=data_test)
expected = target_test

print(predicted[:20])
print(expected[:20])

print(format(knn.score(data_test, target_test), ".2%"))

wrong = [(p,e) for (p,e) in zip(predicted, expected) if p != e]
print(wrong)

from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_true=expected, y_pred=predicted)
print(confusion)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt2

confusion_df = pd.DataFrame(confusion, index=range(10), columns=range(10))
figure = plt2.figure(figsize=(7,6))
axes = sns.heatmap(confusion_df, annot=True, cmap=plt2.cm.nipy_spectral_r)

plt2.show()