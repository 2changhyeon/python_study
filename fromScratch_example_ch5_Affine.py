# -*- coding: utf-8 -*-
"""
Created on Mon May 24 23:54:08 2021

@author: 이창현
"""

import numpy as np

x = np.random.rand(2) #입력
w = np.random.rand(2,3) #가중치
b = np.random.rand(3) #편향

x.shape
w.shape
b.shape

y = np.dot(x,w)+b
print(y)

class Affine: # 행렬 내적 처리 
    def __init__(self,w,b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        out = np.dot(x,self.w)+self.b
        
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        
        return dx
    
    
        
        