# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:03:13 2017

@author: leon
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import seaborn as sns
import matplotlib.pyplot as plt

def datamake():
    train_df = pd.read_csv("pimatr.csv")
    test_df = pd.read_csv("pimate.csv")
    print(train_df.head())
    
    sns.lmplot('glu','bmi',data = test_df, hue = 'type', fit_reg=False)
    plt.title("test data")
    
    sns.lmplot('glu','bmi',data = train_df, hue = 'type', fit_reg=False)
    plt.title("train data")
    
    return train_df, test_df

def calculateStatistics(train_df):    
    #クラス別にデータを準備
    class2 = train_df[train_df.type == 'Yes'][['glu', 'bmi']] #typeがYesのクラス
    class1 = train_df[train_df.type == 'No'][['glu', 'bmi']]  #typeがNoのクラス
    class0 = train_df[['glu', 'bmi']]  #クラス1とクラス2の統合
    
    #各クラスの平均ベクトル
    mu1 = class1.mean()
    mu2 = class2.mean()
    
    #クラスの事前確率
    P1 = len(class1.index) * 1.0 / len(class0.index)
    P2 = len(class2.index) * 1.0 / len(class0.index)
    
    #各クラスの分散共分散行列
    cov1 = class1.cov()
    cov2 = class2.cov()
    
    S = np.linalg.inv(cov1) - np.linalg.inv(cov2)
    c = mu2.dot(np.linalg.inv(cov2)) - mu1.dot(np.linalg.inv(cov1))
    F = ( mu1.transpose().dot(np.linalg.inv(cov1)).dot(mu1.transpose()) 
        - mu2.dot(np.linalg.inv(cov2)).dot(mu2.transpose()) 
        + np.log(np.linalg.det(cov1) / np.linalg.det(cov2)) - 2 * np.log(P1/P2))
                  
    return S, c, F, class0

def calculateDiscriminantFunctionValue(S, c, F, class0, test_df):
    class0 = class0.transpose()
    f1 = np.zeros(len(class0.columns))
    for i in range(len(class0.columns)):
        f1[i] = class0[i].transpose().dot(S).dot(class0[i]) + 2 * c.dot(class0[i]) + F 

    test_df = test_df[['glu', 'bmi']]
    test_df = test_df.transpose()
    f2 = np.zeros(len(test_df.columns))
    for i in range(len(test_df.columns)):
        f2[i] = test_df[i].transpose().dot(S).dot(test_df[i]) + 2 * c.dot(test_df[i]) + F 


    x = np.arange(50,210,1)
    y = np.arange(15,50,1)
    X,Y=np.meshgrid(x,y)
    
    #Xnew Ynewをベクトルに変換
    Xnew = X.reshape(1,35*160)
    Ynew = Y.reshape(1,35*160)
    
    #XYの各列に、XY平面上の点が格納されているイメージ
    XY = np.array([Xnew[0,:],Ynew[0,:]])
    f3 = np.zeros(len(Xnew[0,:]))
    for i in range(len(Xnew[0,:])):
        f3[i] = XY[:,i].transpose().dot(S).dot(XY[:,i]) + 2 * c.dot(XY[:,i]) + F
    
    Z = np.reshape(f3,(35,160))
    plt.contour(X,Y, Z)
    
    return f1, f2

def performanceEvaluate(train_df, test_df, f1, f2):
    N1 = len(train_df.index)
    N2 = len(test_df.index)
    
    train_df['TF'] =DataFrame(f1.reshape(N1, 1))
    train_df['estimateclass'] = train_df['TF'].apply(lambda r: 'Yes' if(r > 0) else 'No')
    sns.lmplot('glu','bmi',data = train_df, hue = 'estimateclass', fit_reg=False)
    plt.title("train data result")
    train_df['correct'] = train_df.apply(lambda r: 1 if(r.type == r.estimateclass) else 0, axis = 1)
    
    test_df['TF'] =DataFrame(f2.reshape(N2, 1))
    test_df['estimateclass'] = test_df['TF'].apply(lambda r: 'Yes' if(r > 0) else 'No')
    sns.lmplot('glu','bmi',data = test_df, hue = 'estimateclass', fit_reg=False)
    plt.title("test data result")
    test_df['correct'] = test_df.apply(lambda r: 1 if(r.type == r.estimateclass) else 0, axis = 1)
    
    
    print('Accuracy(train) : {}'.format(train_df['correct'].sum() / len(train_df.index)))
    print('Accuracy(test) : {}'.format(test_df['correct'].sum() / len(test_df.index)))
  
if __name__ == "__main__":
    train_df, test_df = datamake()
    
    S, c, F, class0 = calculateStatistics(train_df)
    f1, f2 = calculateDiscriminantFunctionValue(S, c, F, class0, test_df)    
    performanceEvaluate(train_df, test_df, f1, f2)
    