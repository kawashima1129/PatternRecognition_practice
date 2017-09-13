# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 12:48:11 2017

@author: Katsuaki Kawashima
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from pandas import DataFrame, Series
import pandas as pd


class GaussianMixture(object):
    def __init__(self, K, input_dim, X):
        #パラメータの初期化
        #この初期値の決め方がなかなか重要になるっぽい　適当にしすぎると失敗しやすくなる
        self.K = K
        self.ndim = input_dim
        
        self.p = np.ones(self.K) / self.K
        self.mu = np.random.uniform(X.min(), X.max(), (self.ndim, self.K))
        self.covs = np.repeat(X.std(axis = 0).mean() * np.eye(self.ndim), self.K).reshape(self.ndim, self.ndim, self.K)
        
    def fit(self, X, max_iter = 100):
        # EM algorithm
        for i in range(max_iter):
            params = np.hstack((self.p.ravel(), self.mu.ravel(), self.covs.ravel()))
            burden_rates = self.e_step(X)#Eステップ
            self.m_step(X, burden_rates)#Mステップ
            #print('iter:{}'.format(i))
            
            #パラメータが変わらなくなったらループを抜ける
            if np.allclose(params, np.hstack((self.p.ravel(), self.mu.ravel(), self.covs.ravel()))):
                break
          
    def gaussian(self, x, k):
        
        x_mu = (x - self.mu[:,k]).reshape(self.ndim, 1)
        p1 =  ( (2 * np.pi) ** (1 / self.ndim) ) * ( np.linalg.det(self.covs[:,:,k]) ** 0.5)  
        p2 = -0.5 * (x_mu).T.dot( np.linalg.inv(self.covs[:,:,k]) ) .dot( x_mu ) 
        p = np.exp(p2) / p1
        
        return p
    
    def e_step(self, X):
        n = np.zeros((len(X),self.K))
        
        for i in range(len(X)):
            d = 0
            for k in range(self.K):    
                n[i, k] = self.p[k] * self.gaussian(X[i,:], k)
                d +=  n[i, k]

            for k in range(self.K):
                n[i, k] /=  d
       
        return n

    def m_step(self, X, burden_rates):   
        Nk = np.sum(burden_rates, axis = 0)
        self.p = Nk / len(X)

        #平均ベクトル算出                
        for k in range(self.K):
            sum_value = np.zeros((1, self.ndim)) 
            for i in range(len(X)):
                sum_value += burden_rates[i,k] * X[i,:] 
            self.mu[:,k] = sum_value / Nk[k]
        
        
        #共分散行列算出
        for k in range(self.K):
            sigma = np.zeros((self.ndim,self.ndim))
            for i in range(len(X)):
                x_mu = (X[i, :] - self.mu[:,k])
                x_mu = x_mu.reshape(self.ndim, 1)
                sigma += burden_rates[i,k] * x_mu.dot(x_mu.T ) 
            self.covs[:,:,k] = ( sigma / Nk[k])
    
    def classify(self, X):
        labels = np.array([np.array([ self.p[k] * self.gaussian(X[i,:], k) for k in range(self.K)]).argmax() 
                            for i in range(len(X))] )     
        return labels
    
    def predict_prob(self, X):
        labels = np.array([np.array([ self.p[k] * self.gaussian(X[i,:], k) for k in range(self.K)]).sum() 
                            for i in range(len(X))] )     
        return labels
        

def create_toy_data():
    x1 = np.random.normal(size=(100, 2))
    x1 += np.array([-5, -5])
    x2 = np.random.normal(size=(100, 2))
    x2 += np.array([5, -5])
    x3 = np.random.normal(size=(100, 2))
    x3 += np.array([0, 5])
    
    return np.vstack((x1, x2, x3))


if __name__ == '__main__':    
    K = 3 #ガウス分布の個数
    
    iris = load_iris()
    df = DataFrame(iris.data[:,2:4]) 
    m = df.mean()
    s = df.std()
    df = df.sub(m).div(s)
    df['y'] = DataFrame(iris.target)
    
    #X =np.array(df[[0,1]])
    X = create_toy_data()
    input_dim = np.size(X,1) #入力次元数              
    gs = GaussianMixture(K, input_dim, X)
    gs.fit(X)
    print("predict:\n u={0}, \nsigma={1}, \npi={2}".format(gs.mu, gs.covs.T, gs.p)) 
    
    labels1 = gs.classify(X) 
    colors = ['red', 'blue', 'green']
    plt.scatter(X[:,0], X[:,1], c = [colors[int(label)] for label in labels1])
    
    x_test, y_test = np.meshgrid(np.linspace( X[:,0].min(), X[:,0].max(), 100),
                                 np.linspace( X[:,0].min(), X[:,0].max(), 100)) 
    lebels2 = gs.predict_prob(np.c_[x_test.ravel(), y_test.ravel()])
    probs = lebels2.reshape(100, 100)
    plt.contour(x_test, y_test, probs)
    