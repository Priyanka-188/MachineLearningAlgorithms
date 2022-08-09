import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

# # print(diabetes.keys())
# print(diabetes.DESCR)
diabetes_X = diabetes.data[:,np.newaxis,2]          #slicing the data to get only one column with all rows

# print(diabetes_X)
diabetes_X_train = diabetes_X[:-30]     #feature for training data
diabetes_X_test = diabetes_X[-30:]    #feature of test data

diabetes_Y_train = diabetes.target[:-30]     #label of training data 
diabetes_Y_test = diabetes.target[-30:]      #label of training data 


model = linear_model.LinearRegression()     #making a model using linear Regression function
model.fit(diabetes_X_train,diabetes_Y_train)      # train the model by fitting the feature of training data and label of training data 
diabeties_Y_predicted = model.predict(diabetes_X_test)   # now, (testing) taking the predicted label for test data using predict function 

print("The mean squared error is : ",mean_squared_error(diabetes_Y_test,diabeties_Y_predicted))

print("Weights : ", model.coef_)
print("Intercept",model.intercept_)

plt.scatter(diabetes_X_test,diabetes_Y_test)    #plotting the test data on the scatter plot
plt.plot(diabetes_X_test,diabeties_Y_predicted)  # making the plot of feature of testing data with predicted label.
plt.show()