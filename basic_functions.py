#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:13:07 2020

@author: xiangyuzhang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:09:04 2020

@author: xiangyuzhang
"""
import pandas as pd
import numpy as np
# import xgboost as xgb
# from NewPredictionModel_AllFeature import predictionModel
from NewPredictionModel_AllFeature import attribute_density
from NewPredictionModel_AllFeature import data_input, attribute_range, relationship_range
from NewPredictionModel_AllFeature import cost_sampled
from scipy.optimize import brute
from scipy import optimize
import timeit
import matplotlib.pylab as plt
from itertools import product
# import pickle
from sklearn.model_selection import train_test_split
from logistic_regression import coef
from logistic_regression import my_cols
import sys  # Need to have acces to sys.stdout
from matplotlib.ticker import PercentFormatter
from matplotlib import colors


def ratio(ele):
    return ele[0]


coef[0] = - np.abs(coef[0]*25)
coef = coef/10
coef[0] = coef[0]*10
coef[29] = 8.7 + 6 + 8*0.7
coef[28] = 6.7 + 6 + 8*0.7
coef[27] = 0.8 + 6 + 8*0.7
coef[26] = - 2.2 + 6 + 8*0.7
coef[25] = - 4.2 + 6 + 8*0.7

RelCus_W=np.array([[0.2, 0.7,  0.05, 0.05, 0],
                   [0.1, 0.1,  0.5,  0.25, 0.05],
                   [0.05,0.05, 0.1,  0.5,  0.3],
                   [0,   0.05, 0.05, 0.1,  0.8],
                   [0,   0,    0,    0.1,  0.9]])
RelCus_L=np.array([[0.9,0.1,0,0,0],
                   [0.1,0.8,0.1,0,0],
                   [0.05,0.05,0.8,0.1,0],
                   [0,0.05,0.05,0.8,0.1],
                   [0,0,0,0.3,0.7]])


def pred(p, relaitionship, *att):
    dim = len(attribute_range)
    l = np.array([p])
    #print(type(att))
    for i in range(dim):
        vec = np.zeros(attribute_range[i] + 1)
        #print(vec)
        vec[att[i]] = 1.
        l = np.append(l, vec)
    vec = np.zeros(relationship_range + 1)
    vec[relaitionship] = 1.0
    l = np.append(l, vec)
    #print(l)
    return 1./(1. + np.exp(-np.dot(l, coef)))


def expected_loss(p, *para):
    *att,relationship, cost = para
    return -(p - cost)*pred(p, relationship, *att)

def total_loss(p, *para):
    *att, relationship, cost, V, gamma, RelCus_W, RelCus_L = para
    winning_probability = pred(p, relationship, *att)
    future_loss = - gamma*(winning_probability*np.dot(RelCus_W[relationship], V)+(1-winning_probability)*np.dot(RelCus_L[relationship], V))
    return expected_loss(p, *att, relationship, cost) + future_loss
