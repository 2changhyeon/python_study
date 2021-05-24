# -*- coding: utf-8 -*-
"""
Created on Mon May 24 23:43:04 2021

@author: 이창현
"""
import numpy as np

class Sigmoid:
    def __init__(self):
        self.out= None
    
    def forward(self, x):
        out = 1/(1+np.exp(-x))
        return out
    
    def backward(self,dout):
        dx = dout *( 1.0-self.out )* self.out
        
        return dx
    