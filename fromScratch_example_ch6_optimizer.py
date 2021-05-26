# -*- coding: utf-8 -*-
"""
Created on Wed May 26 22:38:24 2021

@author: 이창현
"""

import numpy as np

class SGD:
    """확률적 경사 하강법 Stochastic gradient descent"""
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
            

class Momentum:
    """모멘텁을 이용한 SGD"""
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] ## 처음엔 v[key] = 0
            params[key] += self.v[key]
            
            """params = params + momentum*v[key] - lr*grads[key]"""
            
            
class AdaGrad:
    """ 과거의 기울기를 제곱하여 계속 더해간다. 학습을 진행할수록 학습의 갱신강도가 줄어든다."""
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            
            for key, val in params.item():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] += grads[key] * grads[key] 
            """기울기 값을 제곱하여 h를 업데이트 한다"""
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) +1e-7) # 작은값 1e-7을 더하여 h값이 0이 되어도 0이 분모가 되는 경우를 막는다.
            
            
            
    
            
