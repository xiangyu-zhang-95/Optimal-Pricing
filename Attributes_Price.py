#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:09:04 2020

@author: xiangyuzhang
"""
import pandas as pd
import numpy as np
import timeit
import matplotlib.pylab as plt
from itertools import product
import sys
from matplotlib.ticker import PercentFormatter
from matplotlib import colors
from basic_functions import ratio, coef, RelCus_W, RelCus_L, pred, expected_loss, total_loss
from NewPredictionModel_AllFeature import cost_sampled
from NewPredictionModel_AllFeature import data_input, attribute_range, relationship_range
from scipy.optimize import brute

def deal_attribute_difference(V, gamma, deal_attribute, costSample = 20):
    opt = np.zeros((4, 5))
    for q in range(costSample):
        value = cost_sampled(0)
        for index in range(4):
            att = deal_attribute[index]
            for relationship in range(relationship_range + 1):
                cost = value
                para_opt = (*att, relationship, cost, V, gamma, RelCus_W, RelCus_L)
                rranges = ((0.8*cost, 1.5*cost),)
                res_opt = brute(total_loss, rranges, args = para_opt, Ns = 600, full_output=True, finish=None)
                opt[index][relationship] = opt[index][relationship] + res_opt[0]
    opt = opt/costSample
    return opt

def plotWinningProbabilityCurve(deal_list, plotRange):
    low, high = plotRange;
    x = np.arange(low, high, (high - low)/1000.0)
    #print(x)
    y0 = np.zeros(x.size)
    y1 = np.zeros(x.size)
    y2 = np.zeros(x.size)
    y3 = np.zeros(x.size)
    for i in range(x.size):
        y0[i] = pred(x[i], 0, *deal_list[0])
        y1[i] = pred(x[i], 0, *deal_list[1])
        y2[i] = pred(x[i], 0, *deal_list[2])
        y3[i] = pred(x[i], 0, *deal_list[3])
    plt.plot(x, y0, color = 'b', label = 'Deal 1(Hardest-to-Wim)')
    plt.plot(x, y1, color = 'r', label = 'Deal 2')
    plt.plot(x, y2, color = 'y', label = 'Deal 3')
    plt.plot(x, y3, color = 'g', label = 'Deal 4(Easiest-to-Win)')
    plt.xlabel("Price ($M)")
    plt.ylabel("Winning Probability")
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0., fontsize = 8)
    plt.title("Winning Probability Sensitivity")
    plt.show()

V = [20.33175597, 35.01809372, 47.18704856, 58.15459068, 60.3039232]
deal_list = [[0, 6, 2, 7],
             [0, 0, 1, 1],
             [1, 4, 4, 4],
             [1, 7, 4, 8]]

price = deal_attribute_difference(V, 0.9, deal_list)
fig, ax = plt.subplots()

plt.plot([0,1,2,3,4], price[0], 'k^:', color = 'b', label = 'Deal 1(Hardest-to-Win)')
plt.plot([0,1,2,3,4], price[1], 'k^:', color = 'r', label = 'Deal 2')
plt.plot([0,1,2,3,4], price[2], 'k^:', color = 'y', label = 'Deal 3')
plt.plot([0,1,2,3,4], price[3], 'k^:', color = 'g', label = 'Deal 4(Easiest-to-Win)')

plt.xlabel("Client Relationship")
plt.ylabel("Price ($M)")
plt.title("Price Depending on Deal Attribute")

ax.set_xticks(range(5))

plt.legend(bbox_to_anchor=(1.0, 0), loc=4, borderaxespad=0., fontsize = 8)

plt.show()

plotWinningProbabilityCurve(deal_list, (0, 25))
