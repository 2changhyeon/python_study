# -*- coding: utf-8 -*-
"""
Created on Mon May 24 23:36:05 2021

@author: 이창현
"""

class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self,x):
        self.mask = (x <= 0) # x가 0 이하면 true, 아니면 false
        out = x.copy() # 0초과이면 그대로 copy
        out[self.mask] = 0 # 0 나감
        
        return out
    
    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout # dx/dout = 1
        
        return dx

x = np.array([[1.0, -0.5],[-2.0, 3.0]])
print(x)

mask = (x<=0)
print(mask)
