# -*- coding: utf-8 -*-
"""
Created on Fri May 28 15:50:31 2021

@author: 이창현
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def Relu(x):
    return np.maximum(0,x)

def tanh(x):
    return np.tanh(x)

input_data = np.random.randn(1000,100)
node_num = 100
hidden_layer_size = 5
activations = {}

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    """기울기 소실이나, 활성화값이 치우쳐 표현력이 저하되지 않게 하기 위해 w의 표준편차를 다양하게 바꿔보았습니다."""
    # w = np.random.randn(node_num, node_num) * 1
    
    # w = np.random.randn(node_num, node_num) * 0.01
    
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
    
    
    a = np.dot(x,w)
    """활성화 함수로서는 원점대칭인 것이 바람직하기 때문에, sigmoid보다 tanh를 사용하는 것이 더 효과적이다."""
    z = sigmoid(a)
    #z = tanh(a)
    
    activations[i]=z
    """ Relu를 사용할땐 He초기값을 사용하고 sigmoid나 tanh같은 s자 모양 곡선일때는 Xavier 초기값을 쓰는 경우가 많다."""
for i, a in activations.items():
    plt.subplot(1,len(activations), i+1)
    plt.title(str(i+1)+ "-layer")
    plt.hist(a.flatten(), 30, range=(0,1))

plt.show()        