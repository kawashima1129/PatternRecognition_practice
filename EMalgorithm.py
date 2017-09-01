# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 12:48:11 2017

@author: Katsuaki Kawashima
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import seaborn as sns
import math
from scipy import stats

fig = plt.figure(figsize = (12, 4))
ax1 = plt.subplot2grid((1,2), (0,0))
ax2 = plt.subplot2grid((1,2), (0,1))

def gaussian(x, mu, sigma):
    p = math.exp(- pow(x - mu, 2) / (2 * sigma)) / math.sqrt(2 * math.pi * sigma)
    return p

def e_step(df, mu, sigma, p):
    burden_rates = []
    for x in df:
        d = (1 - p) * gaussian(x, mu[0], sigma[0]) + p * gaussian(x, mu[1], sigma[1])
        n = p * gaussian(x, mu[1], sigma[1])
        burden_rate = n / d
        burden_rates.append(burden_rate)
        
    return burden_rates

def m_step(df, burden_rates):
    d = sum(burden_rates)
    n = sum([r * x for x, r in zip(df, burden_rates)])
    mu1 = n / d
    
    n = sum([r * pow(x-mu1, 2) for x, r in zip(df, burden_rates)])
    var1 = n / d
    
    d = sum([(1-r) for r in burden_rates])
    n = sum([(1-r) * x for x, r in zip(df, burden_rates)])
    mu2 = n / d
    
    n = sum([(1-r) * pow(x - mu2, 2) for x, r in zip(df, burden_rates)])
    var2 = n / d
    
    N = len(df)
    p = sum(burden_rates) / N
    
    return np.array([mu1, mu2]), np.array([var1, var2]), p

def calc_log_likelihood(df, mu, sigma, p):
    s = 0
    for x in df:
        g1 = gaussian(x, mu[0], sigma[0])
        g2 = gaussian(x, mu[1], sigma[1])
        s += math.log(p * g1 + (1-p) * g2)
    return s

if __name__ == '__main__':    
    K = 2 #クラス数
    df = pd.read_csv("faithful.csv", header = None, names = ['x1', 'x2'])
    sns.distplot(df.x1, kde = False, bins = 20, norm_hist=True)
    
    #initial gmm parameter
    p = 0.5
    mu = np.random.randn(K)
    sigma = np.abs(np.random.randn(K))
    delta = None
    
    #終了条件
    epsilon = 0.000001
    T = 20
    Q = -sys.float_info.max
    delta = None
    
    # EM algorithm
    ls = []
    #while delta >= epsilon:
    while delta == None or delta >= epsilon:
        burden_rates = e_step(df.x1, mu, sigma, p)
        mu, sigma, p = m_step(df.x1, burden_rates)
        Q_new = calc_log_likelihood(df.x1, mu, sigma, p)
        ls.append(Q_new)
        delta = Q_new - Q
        Q = Q_new
        
        
    print("predict: mu1={0}, mu2={1}, v1={2}, v2={3}, p={4}".format(
        mu[0], mu[1], sigma[0], sigma[1], p)) 
    
    xs = np.linspace(min(df.x1), max(df.x1), 200)
    norm1 = stats.norm.pdf(xs, mu[0], math.sqrt(sigma[0]))
    norm2 = stats.norm.pdf(xs, mu[1], math.sqrt(sigma[1]))
    
    plt.plot(xs, p * norm1 + (1-p) * norm2, color="red", lw=3)
    plt.xlabel('x')
    plt.ylabel('probability')
    
    T = [t for t in range(len(ls))]
    ax1.plot(T, ls)
    ax1.set_xlabel('step')
    ax1.set_ylabel('log_likelihood')

    


