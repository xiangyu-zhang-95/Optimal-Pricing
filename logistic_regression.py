# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:38:41 2018

@author: XiangyuZhang
"""


import pandas as pd
import statistics
import os
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import matplotlib
import numbers
import decimal
import sys
import ast
import numpy as np
from scipy import interp
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


resultsList=[]
resultsList.append(["Classifier Name","Accuracy", "Mean Accuracy", "Standard Deviation of Accuracy","Precision", "Recall","Area Under Curve", "F1 Score"])


FILE_NAME="Input_df_4.xlsx"

data=pd.read_excel(FILE_NAME,index_col=False)
# data.columns = ['Country of the Client', 'Industrial Sector of the Client','How was the Deal Identified', 'Type of the Client', 'Relationship with the Client', 'Value (M$)', 'Outcome']

my_cols = list(data.columns)
my_cols.pop()
my_cols.pop()
data=pd.get_dummies(data,columns=my_cols)

Features=data.drop(["Outcome"],axis=1)

X_train, X_test, y_train, y_test = train_test_split(Features, data["Outcome"],train_size=0.8,test_size=0.2,stratify=data["Outcome"], random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(Features, data["Outcome"],train_size=0.8,test_size=0.2,stratify=data["Outcome"], random_state=42)
#stratify makes sure that the imbalance in the whole dataset is equal to that in both the training and testing dataset splits

param_grid = {"C":10.**np.arange(-1,1)}   
#Note the use of 10. not 10 because even in python 3, 
#the exponential power is integer unless the base is float, so 10**-3 is actually zero 
# (and that wouldn't work for the C parameter of logistic regression)

grid_search = GridSearchCV(LogisticRegression(solver='lbfgs'), param_grid, cv=10)
grid_search.fit(X_train, y_train)
grid_search.predict(X_test)
# print("Logistic Regression Results")
# print(grid_search.score(X_test, y_test))
# print("\n")
LogisticRegressionResults=grid_search.score(X_test, y_test)
#print(grid_search.best_params_)
y_score=  grid_search.predict(X_test)
myClassifier="Logistic Regression"
#Calculate Accuracy, Precision, Recall, AUC, F1_score
myAccuracy=grid_search.score(X_test, y_test)
fpr,tpr,thresholds=roc_curve(y_test,y_score)
myAUC = auc(fpr, tpr)
myRecall=recall_score(y_test,y_score,pos_label=1,average='binary')
myPrecision=precision_score(y_test,y_score,pos_label=1,average='binary')
myF1Score=f1_score(y_test,y_score,pos_label=1,average='binary')

scores=cross_val_score(grid_search,Features,data["Outcome"],cv=10)
#print(scores.mean())
#print(scores.std())
resultsList.append([myClassifier,myAccuracy,scores.mean(),scores.std(),myPrecision,myRecall,myAUC,myF1Score])

# print("Logistic Regression Results", ",",LogisticRegressionResults)

est = grid_search.best_estimator_
varNames = list(Features.columns)
for i in range(len(varNames)):
    print(varNames[i] + ":" + str(est.coef_[0, i]))

coef = est.coef_[0]

#X_test.index
#grid_search.predict([X_test.loc[X_test.index[0]]])























