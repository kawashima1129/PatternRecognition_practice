# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:20:04 2017

@author: Katsuaki Kawashima
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from sklearn.datasets import make_blobs

plt.style.use('ggplot')

fig = plt.figure(figsize = (10, 8))
ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)
ax2 = plt.subplot2grid((2,2), (1,0))
ax3 = plt.subplot2grid((2,2), (1,1))

def data_make(N, draw_plot):
    X, Y = make_blobs(n_samples = 500, centers = 2, cluster_std = 3.0)
    X = pd.DataFrame(X, columns=['x1', 'x2'])
    m = X.mean()
    s = X.std()
    X = X.sub(m).div(s)#標準化
    Y = pd.DataFrame(Y, columns=['y'])
    df_data = pd.concat([X, Y], axis = 1)
    print(df_data.head())
    
    if draw_plot:
        ax2.scatter(x = df_data.x1, y = df_data.x2, c = df_data.y, alpha = 0.5)
        ax2.set_xlim([df_data.x1.min() -0.1, df_data.x1.max() +0.1])
        ax2.set_ylim([df_data.x2.min() -0.1, df_data.x2.max() +0.1])
    
    return df_data

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def estimate_weight(df_data, initial_param, N): 
    #確率的勾配降下法 
    error = []
    w = initial_param
    eta = 0.1 #学習率の初期値
    L = 10#学習回数
    for i in range(L):        
        data_list =[k for k in range(N)]
        np.random.shuffle(data_list)
        
        for n in data_list:
            x = df_data.ix[n, 0]
            y = df_data.ix[n, 1]
            t = df_data.ix[n, 2]
            feature_vector = np.array([x, y, 1])
            z = np.inner(feature_vector, weight_vector) #ベクトルの内積
            predict = (sigmoid(z))
            w = w - eta * (predict - t) * feature_vector
        eta *= 0.9
        error.append(np.abs(predict - t))
    ax1.set_title('error')
    ax1.plot(range(L), error)
    return w

def draw_split_line(weight_vector):
    a, b, c = weight_vector
    x = np.array(range(-10, 10, 1))
    y = (a*x+c)/-b
    ax2.plot(x,y,alpha=0.3)
    ax2.set_title('plot with split line before/after optimization', size=16)
    

def validate_prediction(df_data, weight_vector):
    a, b,c = weight_vector
    df_data['prob'] = df_data.apply(lambda row: 1 if(a*row.x1 + b*row.x2 + c) > 0 else 0, axis = 1)
    df_data['p'] = df_data.apply(lambda row: sigmoid(a*row.x1 + b*row.x2 + c), axis = 1)
    
    return df_data

def draw_prob(df_data, weight_vector):
    df = validate_prediction(df_data, weight_vector)
    z3_plot = ax3.scatter(df_data.x1, df_data.x2, c=df_data.p, cmap='Blues', alpha=0.6)
    ax3.set_xlim([df_data.x1.min() -0.1, df.x1.max() +0.1])
    ax3.set_ylim([df_data.x2.min() -0.1, df.x2.max() +0.1])
    plt.colorbar(z3_plot, ax = ax3)
    ax3.set_title('plot colored by probability', size=16)

if __name__ == "__main__":
    N = 500g
    df_data = data_make(N, draw_plot=True)

    weight_vector = np.random.rand(3)
    weight_vector = estimate_weight(df_data, weight_vector, N)
    print('重みパラメータ : {}'.format(weight_vector))
    
    draw_split_line(weight_vector)
    df_pred = validate_prediction(df_data, weight_vector)
    
    draw_prob(df_data, weight_vector)
    
    df_pred['TF'] = df_pred.apply(lambda r: 1 if(r.y == r.prob) else 0, axis =1)
    print('Accuracy : {}'.format(df_pred['TF'].sum() / N))
     