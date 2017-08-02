# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:20:04 2017

@author: Katsuaki Kawashima
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt

plt.style.use('ggplot')

fig = plt.figure(figsize = (10, 8))
ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)
ax2 = plt.subplot2grid((2,2), (1,0))
ax3 = plt.subplot2grid((2,2), (1,1))

def data_make(N, draw_plot, is_confused, confuse_bin):
    np.random.seed(1)
    
    feature = np.random.randn(N, 2)
    df = pd.DataFrame(feature, columns = ['x', 'y'])
    df['c'] = df.apply(lambda row : 1 if (5*row.x + 3*row.y - 1)>0 else 0,  axis=1)

    if is_confused:
        def get_model_confused(data):
            c = 1 if (data.name % confuse_bin) == 0 else data.c
            return c
        df['c'] = df.apply(get_model_confused, axis = 1)
    
    if draw_plot:
        ax2.scatter(x = df.x, y = df.y, c = df.c, alpha = 0.5)
        ax2.set_xlim([df.x.min() -0.1, df.x.max() +0.1])
        ax2.set_ylim([df.y.min() -0.1, df.y.max() +0.1])
    
    return df

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def estimate_weight(df_data, initial_param, N): 
    #確率的勾配降下法 
    error = []
    w = initial_param
    eta = 0.1 #学習率の初期値
    for i in range(10):        
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
    ax1.plot(range(10), error)
    return w

def draw_split_line(weight_vector):
    a, b, c = weight_vector
    x = np.array(range(-10, 10, 1))
    y = (a*x+c)/-b
    ax2.plot(x,y,alpha=0.3)
    ax2.set_title('plot with split line before/after optimization', size=16)
    

def validate_prediction(df_data, weight_vector):
    a, b,c = weight_vector
    df_data['prob'] = df_data.apply(lambda row: 1 if(a*row.x + b*row.y + c) > 0 else 0, axis = 1)
    df_data['p'] = df_data.apply(lambda row: sigmoid(a*row.x + b*row.y + c), axis = 1)
    
    return df_data

def draw_prob(df_data, weight_vector):
    df = validate_prediction(df_data, weight_vector)
    z3_plot = ax3.scatter(df_data.x, df_data.y, c=df_data.p, cmap='Blues', alpha=0.6)
    ax3.set_xlim([df_data.x.min() -0.1, df.x.max() +0.1])
    ax3.set_ylim([df_data.y.min() -0.1, df.y.max() +0.1])
    plt.colorbar(z3_plot, ax = ax3)
    ax3.set_title('plot colored by probability', size=16)

if __name__ == "__main__":
    N = 500
    df_data = data_make(N, draw_plot=True, is_confused = True, confuse_bin = 10)

    weight_vector = np.random.rand(3)
    weight_vector = estimate_weight(df_data, weight_vector, N)
    print('重みパラメータ : {}'.format(weight_vector))
    
    draw_split_line(weight_vector)
    df_pred = validate_prediction(df_data, weight_vector)
    
    draw_prob(df_data, weight_vector)
    
    df_pred['TF'] = df_pred.apply(lambda r: 1 if(r.c == r.prob) else 0, axis =1)
    print('Accuracy : {}'.format(df_pred['TF'].sum() / N))
     