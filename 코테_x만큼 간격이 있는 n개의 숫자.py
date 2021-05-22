# -*- coding: utf-8 -*-
"""
Created on Fri May 21 22:56:39 2021

@author: 이창현
"""

def solution (x,n):
    number = 0;
    answer = []
    num = 0
    while num < n :
        number += x 
        answer.append(number)
        num += 1
        
    return answer


def number_generator (x,n):
    return[i*x+x for i in range(n)]

print(number_generator(2,5))