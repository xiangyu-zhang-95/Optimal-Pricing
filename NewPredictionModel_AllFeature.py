# -*- coding: utf-8 -*-
"""
Created on Thu May 10 16:23:10 2018

@author: xz556
"""

import numpy as np
#import xgboost as xgb
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd


# precondition for trainfile
# feature = [..., relationship, value, outcome]
trainfile = "Input_df_4.xlsx"
data_input=pd.read_excel(trainfile, index_col=False)

feature = data_input.columns.tolist()
data = data_input.values
attribute_range = [data_input[feature[i]].max() for i in range(len(feature) - 3)]
relationship_range = data_input[feature[len(feature) - 3]].max()

costRange = data_input[feature[len(feature) - 2]].max()

def dataAndfeature():
    data_input=pd.read_excel(trainfile, index_col=False)
    feature = data_input.columns.tolist()
    data = data_input.values
    return (data, feature, data_input)

def predictionModel():
    data, feature, data_input = dataAndfeature()
    data = upSample(data, feature)
    data = data.as_matrix()
    X = data[:, 0 : len(feature) - 1]
    y = data[:,len(feature) - 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    params = {
            'max_depth': 2,
            'eta': 0.1,
            'silent': 1,
            'eval_metric': 'rmse',
            'seed': 154
            }
    
    dtrain = xgb.DMatrix(X_train, label = y_train)
    dvalid = xgb.DMatrix(X_test, label = y_test)
    
    params_constrained = params.copy()
    constains = list()
    for i in range(len(feature) - 3):
        constains.append(0)
    constains.append(1)
    constains.append(-1)
    constains = tuple(constains)
    
    params_constrained['monotone_constraints'] = str(constains)
    
    evallist  = [(dtrain, 'train'), (dvalid, 'eval')]
    model_with_constraints = xgb.train(params_constrained, dtrain, 
                                       num_boost_round = 2500, evals = evallist, 
                                       early_stopping_rounds = 10)
    print(constains)
    return model_with_constraints

def upSample(df, feature):
    length = len(feature)
    dff = pd.DataFrame(df, columns = feature)
    df_0 = dff[dff[feature[length - 1]] == 0]
    df_1 = dff[dff[feature[length - 1]] == 1]
    df_1s = resample(df_1,replace=True,random_state=42,n_samples=len(df_0))
    df_final = pd.concat([df_0,df_1s],axis=0)
    return df_final

def density():
    data=pd.read_excel(trainfile,index_col=False)
    feature = data.columns.tolist()
    density = list()
    for i in range(len(feature) - 3):
        large = data[feature[i]].max()
        a = np.zeros(large + 1)
        for j in range(large+1):
            a[j] = data[data[feature[i]] == j].shape[0]
        a = a/np.sum(a)
        density.append(a)
    return density

def cost_sampled(relationship):
    index = np.random.randint(0, data.shape[0] - 1)
    return data_input[feature[len(feature) - 2]][index] + 14 
    # return data_input[feature[len(feature) - 2]][index]*(0.8 + (1 - (relationship+0.0)/relationship_range)* 0.1)

attribute_density = density()
"""
def plot_one_feature_of_two_prediction(bst, X, y, relationship, color):
    x_scan = np.linspace(0, costRange, 1000)    
    X_scan = np.empty((1000, X.shape[1]))
    for i in range(1000):
        for j in range(X.shape[1]):
            X_scan[i,j] = 0
    for i in range(1000):
            X_scan[i,len(feature)-3] = relationship
    X_scan[:, len(feature)-2] = x_scan
    X_plot = xgb.DMatrix(X_scan)
    y_plot = bst.predict(X_plot)
    plt.plot(x_scan, y_plot, color)

# data from Input_df_4.xlsx
data, feature, data_input = dataAndfeature()
X = data[:, 0:len(feature) - 1]
y = data[:,len(feature) - 1]
X_train, X_test, y_train, y_test = train_test_split(X, y)

model_with_constraints = predictionModel()

plot_one_feature_of_two_prediction(model_with_constraints, X_test, y_test, 0, 'r')
plot_one_feature_of_two_prediction(model_with_constraints, X_test, y_test, 1, 'g')
plot_one_feature_of_two_prediction(model_with_constraints, X_test, y_test, 2, 'b')
plot_one_feature_of_two_prediction(model_with_constraints, X_test, y_test, 3, 'm')
plot_one_feature_of_two_prediction(model_with_constraints, X_test, y_test, 4, 'y')
plt.title("Probabilty")
plt.xlabel("Price")
plt.ylabel("Winning Probability")
"""