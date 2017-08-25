# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 13:54:03 2017

@author: Katsuaki Kawashima
"""

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
from sklearn.datasets import load_iris
import numpy as np

fig = plt.figure(figsize = (12, 4))
ax1 = plt.subplot2grid((1,2), (0,0))
ax2 = plt.subplot2grid((1,2), (0,1))

class KMeans(object):
    def __init__(self, n_clusters, max_iter = 300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cluster_centers_ = None

    def fit_predict(self, df):
        #初期データのラベルはランダム
        pred = np.random.randint(0, self.n_clusters, len(df.index))
        pred_df = DataFrame(pred, columns = ['pred'])
        df = pd.concat([df, pred_df], axis = 1)
       
        for _ in range(self.max_iter):
            #各クラスタごとにセントロイド（平均ベクトル）を計算
            self.cluster_centers_ = np.array([df[['x1', 'x2']][df.pred == i].mean(axis = 0)  for i in range(self.n_clusters)])
            
            #最も近いセントロイドのラベルをつける
            new_pred = np.array([ np.array([self._euclidean_distance(df.ix[p,['x1', 'x2']], centroid) 
            for centroid in self.cluster_centers_]).argmin() for p in range(len(df.index)) ] )

            #ラベルが前回と変わらなかったらループを抜ける
            if np.all(new_pred == pred):
                break
            
            df.pred = new_pred
            pred = new_pred
            
            
        return pred
    
    def _euclidean_distance(self, p0, p1):
        return np.sum((p0 - p1) ** 2)
        
if __name__=="__main__":
    N_CLUSTERS = 3#クラスタ数
    iris = load_iris()
    X = iris.data
    Y = iris.target
    x1 = Series(X[:,2])
    x2 = Series(X[:,3])
    y = Series(Y)
    df = pd.concat([x1, x2, y], axis = 1)
    df.columns = ['x1', 'x2', 'y']
    ax1.scatter(x = df.x1, y = df.x2, c = df.y, alpha = 0.5)
    
    cls = KMeans(N_CLUSTERS)
    pred= cls.fit_predict(df)
    
    plt.scatter(cls.cluster_centers_[:, 0], cls.cluster_centers_[:, 1], s=100,
                facecolors='none', edgecolors='black')
    
    pred_df = DataFrame(pred, columns = ['pred'])
    df = pd.concat([df, pred_df], axis = 1)
    ax2.scatter(x = df.x1, y = df.x2, c = df.pred, alpha = 0.5)
    
    
    