# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:14:06 2017

@author: Katsuaki Kawashima
"""

import numpy as np
from scipy.stats import norm, skewnorm, entropy
import matplotlib.pyplot as plt

"""
#練習
ndiv = 100
px = np.linspace(norm.ppf(0.001), (norm.ppf(0.999)*1.1), ndiv)
pdf_std = norm.pdf(px, loc=0, scale=1)
pdf_ij = norm.pdf(px, loc=0, scale=0.8)
dkl_ij = entropy(pdf_std, pdf_ij)
s=sum( (pdf_std / sum(pdf_std)) * np.log( (pdf_std / sum(pdf_std)) / (pdf_ij /sum(pdf_ij)) ) )#合計が1になるように正規化する
plt.plot(px, pdf_ij)
plt.plot(px, pdf_std)
title_str = 'DKL = {:.5f}'.format(dkl_ij)
plt.title(title_str)
"""

"""
2つのGMMのKL情報量を計算（Monte Carlo Sampling）
参考
LOWER AND UPPER BOUNDS FOR APPROXIMATION OF THE KULLBACK-LEIBLER
DIVERGENCE BETWEEN GAUSSIAN MIXTURE MODELS

"""

class GMM(object):
    def __init__(self, pi, mu, sigma):
        self.pi = pi
        self.mu = mu
        self.sigma = sigma
    
    def gauss(self, x):
        y = [(1/np.sqrt(2*np.pi*sigma)) * (np.exp(-(x-mu)**2 / (2*sigma))) for mu, sigma in zip(self.mu, self.sigma)]
        y = y[0]*self.pi[0] + y[1]*self.pi[1]
        
        return y

class KLDivergence(object):
    def calc(self, gmm1, gmm2, max_iter):
        sum_=0
        
        for _ in range(max_iter):
            ch = np.random.choice(np.array([0, 1]), 1, p=[0.7,0.3])
            x = np.random.normal(gmm1.mu[int(ch)], gmm1.sigma[int(ch)]**0.5)
            
            p = gmm1.gauss(x)
            q = gmm2.gauss(x)
            
            if p==0 or q==0: #確率0が出現すると結果がnanになってしまうため．
                sum_ = sum_+0
            else:
                sum_ = sum_ + np.log(p/q)
        return sum_/max_iter
        
        
if __name__ == '__main__':
    pi1 = [0.7, 0.3]
    pi2 = [0.3, 0.7]
    mu1 = mu2 = [100, 10000]
    sigma1 = sigma2 = [3, 500]
    gmm1 = GMM(pi1, mu1, sigma1)
    gmm2 = GMM(pi2, mu2, sigma2)
    
    kl = KLDivergence()
    result = kl.calc(gmm1, gmm2, 1000)
    print('KL Divergence: {}'.format(result))
    
    
