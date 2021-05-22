# -*- coding: utf-8 -*-
"""
Created on Sat May 22 11:33:34 2021

@author: 이창현
"""
a = 11

def solution (a):
    num = a
    a = str(a)
    sum = 0
    for digit in a:
        sum += int(digit)
            
    return (num % sum)==0

def solution1 (a):
            
    return a % sum([int(c) for c in str(a)])==0



print(solution(10))
print(solution(11))
print(solution(12))
print(solution(13))