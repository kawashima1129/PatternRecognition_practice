# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:16:23 2017

@author: Katsuaki Kawashima
"""

import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize = (12, 4))
ax1 = plt.subplot2grid((1,2), (0,0))
ax2 = plt.subplot2grid((1,2), (0,1))

def kernel(x, y):
    return (np.dot(x, y) + 1) ** 2

                 
def kpca(train_data, N):
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            K[i, j] = K[j, i] = kernel(train_data[i,0:2], train_data[j,0:2])
    ones = np.mat(np.ones((N, N))) / N
    K = K- ones * K - K * ones + ones * K *ones
    return K
                 
                 
if __name__=='__main__':
    train_data, labels = make_moons(n_samples=100, random_state=123)
    
    color = ['r', 'g', 'b']
    train_data = np.loadtxt('kpca.dat')
    N = len(train_data)
    for i in range(N):
        ax1.scatter(train_data[i,0], train_data[i,1], c = color[int(train_data[i,2])])
    
    K = kpca(train_data, N)
    w, v = np.linalg.eig(K)
    ind = np.argsort(w)
    x1 = ind[-1]
    x2 = ind[-2]
    
    for i in range(N):
        ax2.scatter(v[i,x1],v[i,x2],  c = color[int(train_data[i,2])])
    
    