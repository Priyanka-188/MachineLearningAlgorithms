# Train a logistic regression classifier to predict wheather a flower is irsi viginica or not .
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
# print(iris.keys())
# print(iris['data'].shape)
print(iris['DESCR'])

X = iris['data'][:,3:]
Y =( iris['target']==2)

#Train a logisitic regression classifier
clf = LogisticRegression()
clf.fit(X,Y)

#predicting the label
example = clf.predict([[2.26]])
# print("The output will be True if the flower is iris else it will be False.")
if example:
    print("Yes the flower is Iris.")
else: 
    print("THe flower is not Iris.")


#printing the actual percentage and using the matplotlib , ploting the logistice regression .
X_new = np.linspace(0,3,1000).reshape(-1,1)
y_probability = clf.predict_proba(X_new)
print(y_probability)

plt.plot(X_new,y_probability[:,1],"g-",label = 'Virginica')
plt.legend()
plt.show()
