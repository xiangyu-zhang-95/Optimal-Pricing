# -*- coding: utf-8 -*-
"""
Created on Sat May 12 18:05:52 2018

@author: xz556
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
for i in range(1, 24):
	coef[i] = coef[i]*20

print(coef)
RelCus_W=np.array([[0.2, 0.7,  0.05, 0.05, 0],
                   [0.1, 0.1,  0.5,  0.25, 0.05],
                   [0.05,0.05, 0.1,  0.5,  0.3],
                   [0,   0.05, 0.05, 0.1,  0.8],
                   [0,   0,    0,    0.1,  0.9]])
RelCus_L=np.array([[0.9,0.1,0,0,0],
                   [0.2,0.8,0.,0,0],
                   [0.05,0.05,0.8,0.1,0],
                   [0,0.05,0.05,0.8,0.1],
                   [0,0,0,0.3,0.7]])

# return winning probability
# input: price, relaitionship, att is tuple of iid deal attributes
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


# return expected loss in current deal
# input: price, para is tuple of iid deal attributes + relationship + cost
def expected_loss(p, *para):
    *att,relationship, cost = para
    return -(p - cost)*pred(p, relationship, *att)

def total_loss(p, *para):
    *att, relationship, cost, V, gamma, RelCus_W, RelCus_L = para
    winning_probability = pred(p, relationship, *att)
    future_loss = - gamma*(winning_probability*np.dot(RelCus_W[relationship], V)+(1-winning_probability)*np.dot(RelCus_L[relationship], V))
    return expected_loss(p, *att, relationship, cost) + future_loss
# total_loss(1, 0,0,0,0, 0, 0.5, np.array([0.,0.,0.,0.,0.]), 0.9, RelCus_W, RelCus_L)

def updateValue(V, W, gamma, Rel_W = RelCus_W, Rel_L = RelCus_L, costSample = 1, lower = 0.8, upper = 1.5):
    MV = np.array([0.,0.,0.,0.,0.])
    expectationV = np.array([0.,0.,0.,0.,0.])
    expectationW = np.array([0.,0.,0.,0.,0.])
    priceV = np.array([0.,0.,0.,0.,0.])
    priceW = np.array([0.,0.,0.,0.,0.])
    iterator = tuple([range(i+1) for i in attribute_range])
    for att in product(*iterator):
        for q in range(costSample):
            value = cost_sampled(0)
            for relationship in range(relationship_range + 1):
                #cost = value
                #para_opt = (*att, relationship, cost, V, gamma, Rel_W, Rel_L)
                #para_myo = (*att, relationship, cost)
                #rranges = ((0.8*cost, 1.5*cost),)
                #res_opt = brute(total_loss, rranges, args = para_opt, Ns = 150, full_output=True, finish=None)
                #res_myo = brute(total_loss, rranges, args = para_myo, Ns = 150, full_output=True, finish=None)
                cost = value
                
                para_opt = (*att, relationship, cost, V, gamma, RelCus_W, RelCus_L)
                para_myo = (*att, relationship, cost, MV, gamma, RelCus_W, RelCus_L)
                rranges = ((lower*cost, upper*cost),)
                res_opt = brute(total_loss, rranges, args = para_opt, Ns = 75, full_output=True, finish=None)
                res_myo = brute(total_loss, rranges, args = para_myo, Ns = 75, full_output=True, finish=None)
                density_list = []
                for i in range(len(attribute_range)):
                    density_list.append(attribute_density[i][att[i]])
                #optimalP[relationship] = optimalP[relationship] + res_opt[0]*np.prod(np.array(density_list))/costSample
                #myopicP[relationship] = myopicP[relationship] + res_myo[0]*np.prod(np.array(density_list))/costSample
                
                density_list = []
                for i in range(len(attribute_range)):
                    density_list.append(attribute_density[i][att[i]])
                priceV[relationship] = priceV[relationship] + res_opt[0]*np.prod(np.array(density_list))/costSample
                priceW[relationship] = priceW[relationship] + res_myo[0]*np.prod(np.array(density_list))/costSample
                expectationV[relationship] = expectationV[relationship] + (-total_loss(res_opt[0], *para_opt))*np.prod(np.array(density_list))/costSample
                expectationW[relationship] = expectationW[relationship] + (-total_loss(res_myo[0], *para_opt))*np.prod(np.array(density_list))/costSample
    return (expectationV, expectationW, priceV, priceW)

def BellmanMap(V, MV, gamma, Rel_W = RelCus_W, Rel_L = RelCus_L, learning_rate = 0.1, lower = 0.8, upper = 1.5):
    W, MW, priceV, priceMV = updateValue(V, MV, gamma, Rel_W, Rel_L, lower = lower, upper = upper)
    # print(W)
    U = (1. - learning_rate)*V + learning_rate*W
    MU = (1. - learning_rate)*MV + learning_rate*MW
    return (U, MU, priceV, priceMV)
#V = np.array([1.,1.,1.,1.,1.])
#BellmanMap(V, 0.9, RelCus_W, RelCus_L, 1)

def Iteration(gamma, step_size = 10, epsilon = 0.001):
    fd = open('nr_6.txt','w') # open the result file in write mode
    old_stdout = sys.stdout
    sys.stdout = fd # Now your file is used by print as destination 
    V = np.array([0.,0.,0.,0.,0.])
    MV = np.array([0.,0.,0.,0.,0.])
    start = timeit.default_timer()
    iteration = 0
    while True:
        U, MU, priceO, priceM = BellmanMap(V, MV, gamma, RelCus_W, RelCus_L, step_size/(step_size + iteration))
        print("Optimal Profit")
        print(U)
        print("Myopic Profit")
        print(MU)
        print("Increased Margin")
        print((U/MU - 1)*100)
        print("Optimal Price")
        print(priceO)
        print("Myopic Price")
        print(priceM)
        print("Discount")
        print((1 - priceO/priceM)*100)
        stop = timeit.default_timer()
        print(stop - start)
        iteration = iteration + 1
        print(iteration)
        print("")
        
        sys.stdout = old_stdout
        print("Optimal Profit")
        print(U)
        print("Myopic Profit")
        print(MU)
        print("Increased Margin")
        print((U/MU - 1)*100)
        print("Optimal Price")
        print(priceO)
        print("Myopic Price")
        print(priceM)
        print("Discount")
        print((1 - priceO/priceM)*100)
        print(stop - start)
        print(iteration)
        print("")
        
        sys.stdout = fd
        if (np.sum(np.abs(U - V)) + np.sum(np.abs(MU - MV)))/(np.sum(U) + np.sum(MU)) < epsilon:
            break;
        V, MV = U, MU
    sys.stdout = old_stdout
    fd.close()
    return (V, MV)

# T = 30
V = [20.33175597, 35.01809372, 47.18704856, 58.15459068, 60.3039232]
MV = [18.38670358, 31.50177085, 45.93845881, 58.15322483, 60.30389772]
print(np.array(V) / np.array(MV) * 100 - 100)
print("V: ", V)
print("MV: ", MV)


def plotValueFunction(V):
    plt.plot(V,'o-')
    plt.ylabel("Value Function")
    plt.xlabel('Client Relationship')
    return

def Price(V, gamma, costSample = 1, lower = 0.8, upper = 1.5):
    myopicP = np.zeros(relationship_range + 1)
    optimalP = np.zeros(relationship_range + 1)
    price_list = list()
    W = np.array([0.,0.,0.,0.,0.])
    iterator = tuple([range(i+1) for i in attribute_range])
    for att in product(*iterator):
        for q in range(costSample):
            value = cost_sampled(0)
            for relationship in range(relationship_range + 1):
                cost = value
                #cost = value*(0.8 + (1 - (relationship+0.0)/relationship_range)* 0.1)
                para_opt = (*att, relationship, cost, V, gamma, RelCus_W, RelCus_L)
                para_myo = (*att, relationship, cost, W, gamma, RelCus_W, RelCus_L)
                rranges = ((lower*cost, upper*cost),)
                res_opt = brute(total_loss, rranges, args = para_opt, Ns = 75, full_output=True, finish=None)
                res_myo = brute(total_loss, rranges, args = para_myo, Ns = 75, full_output=True, finish=None)
                density_list = []
                for i in range(len(attribute_range)):
                    density_list.append(attribute_density[i][att[i]])
                optimalP[relationship] = optimalP[relationship] + res_opt[0]*np.prod(np.array(density_list))/costSample
                myopicP[relationship] = myopicP[relationship] + res_myo[0]*np.prod(np.array(density_list))/costSample
                entry = [(1-res_opt[0]/res_myo[0])*100, res_myo[0], res_opt[0], *att, relationship, cost]
                # print(entry[0])
                price_list.append(entry)
    #price_list.sort(key = ratio)
    return (myopicP, optimalP, price_list)

def plotPrice(myopicP, optimalP):
    plt.plot(myopicP, 'k^:', color='b', label='Myopic Price')
    plt.plot(optimalP, 'k^:', color='r', label='Optimal Price')
    plt.ylabel("Price ($M)")
    plt.xlabel('Firm-Customer Relationship')
    plt.legend(bbox_to_anchor=(1, 0), loc=4, borderaxespad=0., fontsize = 8)
    plt.savefig("provider_client_relationship.png")
    # plt.title("Myopic Price versus Optimal Price")
    return



def cost_price_plot(V, gamma, lower = 0.8, upper = 1.5):
    costs = np.arange(10, 30, 5)
    price = np.zeros((5, len(costs)))
    #att = (0,0,0,0)
    iterator = tuple([range(i+1) for i in attribute_range])
    for att in product(*iterator):
        for relationship in range(5):
            for i in range(len(costs)):
                cost = costs[i]
                para_opt = (*att, relationship, cost, V, gamma, RelCus_W, RelCus_L)
                rranges = ((lower*cost, upper*cost),)
                res_opt = brute(total_loss, rranges, args = para_opt, Ns = 75, full_output=True, finish=None)
                density_list = []
                for j in range(len(attribute_range)):
                    density_list.append(attribute_density[j][att[j]])
                price[relationship, i] += res_opt[0] * np.prod(np.array(density_list))
    print(price)
    fig, ax = plt.subplots()
    plt.plot([0,1,2,3,4], np.array(price[:, 0]), 'k^:', color = 'b', label = 'cost = 10 M')
    plt.plot([0,1,2,3,4], np.array(price[:, 1]), 'k^:', color = 'r', label = 'cost = 15 M')
    plt.plot([0,1,2,3,4], np.array(price[:, 2]), 'k^:', color = 'y', label = 'cost = 20 M')
    plt.plot([0,1,2,3,4], np.array(price[:, 3]), 'k^:', color = 'g', label = 'cost = 25 M')
    
    ax.set_xticks(range(5))
    plt.xlabel("Firm-Customer Relationship")
    plt.ylabel("Optimal Price ($M)")
    plt.legend(bbox_to_anchor=(1.0, 0), loc=4, borderaxespad=0., fontsize = 8)
    plt.savefig("price_depending_on_cost.png")
    plt.show()
            
            
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
    plt.legend(bbox_to_anchor=(1, 0), loc=4, borderaxespad=0., fontsize = 8)
    plt.title("Winning Probability Sensitivity")
    plt.savefig("deal_attribute_probability.png")
    plt.show()


def attribute_price_plot(V):
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
    plt.xlabel("Firm-Customer Relationship")
    plt.ylabel("Price ($M)")
    plt.title("Price Depending on Deal Attribute")
    ax.set_xticks(range(5))
    plt.legend(bbox_to_anchor=(1.0, 0), loc=4, borderaxespad=0., fontsize = 8)
    plt.savefig("deal_attribute_price.png")
    plt.show()
    plotWinningProbabilityCurve(deal_list, (0, 25))

def myo_opt(V, gamma, FILE_NAME):
    W = np.zeros(5);
    data=pd.read_excel(FILE_NAME,index_col=False)
    opt_list = []
    myo_list = []
    discount = []
    num = data.shape[0]
    for i in range(int(num/20)):
        att = (int(data.ix[i][0]), int(data.ix[i][1]), int(data.ix[i][2]), int(data.ix[i][3]))
        relationship = int(data.ix[i][4])
        cost = data.ix[i][5]*5
        para_opt = (*att, relationship, cost, V, gamma, RelCus_W, RelCus_L)
        para_myo = (*att, relationship, cost, W, gamma, RelCus_W, RelCus_L)
        rranges = ((0.0, 10*cost),)
        res_opt = brute(total_loss, rranges, args = para_opt, Ns = 150, full_output=True, finish=None)
        res_myo = brute(total_loss, rranges, args = para_myo, Ns = 150, full_output=True, finish=None)
        opt_list.append(res_opt[0]);
        myo_list.append(res_myo[0]);
        discount.append((1 - res_opt[0]/res_myo[0])*100)
    discount = np.array(discount)
    n_bin = [0, 10, 20, 30, 40, 50]
    fig, ax = plt.subplots()
    # plt.hist(discount, bins = n_bin, normed = True)
    plt.xlabel("Discount(%)")
    plt.ylabel("Percentage")
    plt.title("Discount Histogram")
    ax.set_xticks(range(0, 60, 10))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=0.1))
    # N is the count in each bin, bins is the lower-limit of the bin
    N, bins, patches = ax.hist(discount, bins=n_bin, density = True)
        
    # We'll color code by height, but you could use any scalar
    fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
            
    # we need to normalize the data to 0..1 for the full range of the colormap
    # norm = colors.Normalize(fracs.min(), fracs.max())
    
    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(thisfrac)
        thispatch.set_facecolor(color)
        
        #plt.xlabel("Deal ID")
        #plt.ylabel("Discount")
    plt.savefig("myo_opt.png")
    plt.show()
    return

V = [20.33175597, 35.01809372, 47.18704856, 58.15459068, 60.3039232]
cost_price_plot(V, 0.9, lower = 0.8, upper = 1.5)

"""
V = [20.33175597, 35.01809372, 47.18704856, 58.15459068, 60.3039232]
myo_opt(V, 0.9, "Input_df_4.xlsx")


myo = [18.11388214, 18.47405821, 20.44192256, 24.28926883, 24.39559725]
opt = [13.02570783, 13.70781302, 18.25558533, 24.27421921, 24.39426477]
plotPrice(myo, opt)


V = [20.33175597, 35.01809372, 47.18704856, 58.15459068, 60.3039232]
attribute_price_plot(V)
"""