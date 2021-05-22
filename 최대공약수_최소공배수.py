# -*- coding: utf-8 -*-
"""
Created on Sat May 22 16:07:32 2021

@author: 이창현
"""

def gcdlcm(a, b):
    c, d = max(a, b), min(a, b)
    t = 1
    while t > 0:
        t = c % d
        c, d = d, t
    answer = [c, int(a*b/c)]

    return answer




def solution(n, m):
    Max = n*m
    if n>m:
        Min = m
    else:
        Min = n
    i=1
    res = 1
    for i in range(Min+1,1,-1):
        if n%i==0 and m%i==0:
            res = i
            break    

    answer = [res,Max//res]

    return answer