# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:57:55 2021

@author: 이창현
"""
import numpy as np
import matplotlib.pylab as plt



def function_1(x): # 변수 1개의 함수
    return 0.01*x**2 + 0.1*x


def function_2(x): # 변수 2개의 함수
    return x[0]**2+x[1]**2


def numerical_diff(f,x): #기울기구함
    h = 1e-4
    return (f(x+h)-f(x-h))/(2*h)


def numerical_gradient(f,x): #모든 변수의 편미분을 벡터로 정리한 기울기
    h = 1e-4
    grad = np.zeros_like(x) # x와 형상이 같고 그 원소가 모두 0인 배열을 만든다.
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    
    return grad
        
def gradient_descent(f, init_x, lr=0.01, step_num=100): # f 함수 init_x 초깃값, lr 학습률, step_num 반복횟수
    x = init_x #초기값 지정
    
    for i in range(step_num): # step_num만큼 반복
        grad = numerical_gradient(f,x) #미분값을 받아서
        x -= lr * grad  # x - lr*편미분
    
    return x



x = np.arange(0.0,20.0,0.1)
y = function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
#plt.show()

numerical_diff(function_1,5)

print(numerical_gradient(function_2, np.array([3.0,4.0])),
numerical_gradient(function_2, np.array([0.0,2.0])),
numerical_gradient(function_2, np.array([3.0,0.0])))