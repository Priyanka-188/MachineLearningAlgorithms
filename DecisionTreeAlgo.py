import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

cancer = datasets.load_breast_cancer()
# print(cancer.keys())
# print(cancer['target'])
# print(cancer['data'])
# print(cancer['DESCR'])
# print(cancer['data'].shape)

# define a target column
print(cancer.shape)