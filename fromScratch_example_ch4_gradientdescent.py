# -*- coding: utf-8 -*-
"""
Created on Mon May 24 19:11:17 2021

@author: 이창현
"""

import sys ,os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W=np.random.randn(2,3)
        
    def predict(self,x):
        return np.dot(x, self.W)
    
    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)
        
        return loss
    

net = simpleNet()
print(net.W) # 가중치 매개변수
x = np.array([0.6,0.9])
p = net.predict(x)
print(p)

np.argmax(p) # 최댓값 인덱스

t = np.array([0,0,1]) #정답지
net.loss(x,t)

def f(W):
    return net.loss(x,t)

dW = numerical_gradient(f, net.W)
print(dW)