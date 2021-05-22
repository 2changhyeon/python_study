# -*- coding: utf-8 -*-
"""
Created on Sat May 22 16:26:34 2021

@author: 이창현
"""

def solution(arr):
    i = arr.index(min(arr))
    arr.pop(i)
    return arr or [-1]


print(solution([4,3,2,1]))
print(solution([10]))
    