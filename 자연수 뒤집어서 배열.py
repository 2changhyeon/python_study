# -*- coding: utf-8 -*-
"""
Created on Sat May 22 17:06:46 2021

@author: 이창현
"""

def solution(n):
    return list(map(int, reversed(str(n))))
    

print(solution(54321))