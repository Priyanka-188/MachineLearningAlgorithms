import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

# # print(diabetes.keys())
# print(diabetes.DESCR)
diabetes_X = diabetes.data

# print(diabetes_X)
diabetes_X_train = diabetes_X[:-30]     #feature for training data
diabetes_X_test = diabetes_X[-30:]    #feature of test data

diabetes_Y_train = diabetes.target[:-30]     #label of training data 
diabetes_Y_test = diabetes.target[-30:]      #label of training data 


model = linear_model.LinearRegression()
model.fit(diabetes_X_train,diabetes_Y_train)      # train the model
diabeties_Y_predicted = model.predict(diabetes_X_test)

print("The mean squared error is : ",mean_squared_error(diabetes_Y_test,diabeties_Y_predicted))

print("Weights : ", model.coef_)
print("Intercept",model.intercept_)

