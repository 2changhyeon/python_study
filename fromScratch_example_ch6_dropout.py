# -*- coding: utf-8 -*-
"""
Created on Fri May 28 19:25:59 2021

@author: 이창현
"""
import numpy as np

class DropOut:
    def __init__(self, dropout_ratio = 0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        
    def forward(self, x, train_flg= True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        
        else:
            return x * (1.0 - self.dropout_ratio)