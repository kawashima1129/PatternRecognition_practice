# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 15:24:48 2017
@author: Katsuaki Kawashima
単純パーセプトロン
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
from pandas import DataFrame, Series

    
def estimate_weight(df_data, initial_param):
    w = initial_param
    c = 1 #固定増分誤り訂正法
    
    while True:
        data_list = list(range(100))
        np.random.shuffle(data_list)
        miss = 0
        for n in data_list:
            x1 = df_data.ix[n, 0]
            x2 = df_data.ix[n, 1]
            y = df_data.ix[n, 2]
            feature_vector = np.array([x1, x2, 1])
            
            predict = np.sign((w * feature_vector).sum())#符号関数：正は1.0，負は-1.0，0は0.0を返す
            
            if predict != y:
                w = w + c * y * feature_vector
                miss += 1
            
        if miss == 0:
            break
    
    return w

def drawline(weight_vector):
    a, b, c = weight_vector
    x = np.array(range(int(df_data.x1.min() - 1) , int(df_data.x1.max()) + 1 , 1))
    y = (a*x+c)/-b
    plt.plot(x,y)
    
    
if __name__=='__main__':
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=8, cluster_std=1.5)
    plt.figure(figsize=(10,10))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='jet')
    
    data_df1 = DataFrame(X, columns = ['x1','x2'])
    data_df2= DataFrame(y, columns = ['y'])
    df_data = pd.concat([data_df1, data_df2], axis = 1)
    df_data['y'] = df_data.apply(lambda r: -1 if(r.y == 0) else 1, axis = 1)
    
    weight_vector = np.random.rand(3)
    print("パラメーラ更新前:{}".format(weight_vector))
    
    weight_vector = estimate_weight(df_data, weight_vector)
    print("パラメーラ更新後:{}".format(weight_vector)) 
    drawline(weight_vector)
    
    
    
    
    
   