# -*- coding: utf-8 -*-
"""
Created on Sat May 22 16:53:21 2021

@author: 이창현
"""

def solution(n):
    n = list(str(n))
    n.sort(reverse=True)
    
    return int("".join(n))

print(solution(118372))
