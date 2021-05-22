# -*- coding: utf-8 -*-
"""
Created on Sat May 22 17:53:37 2021

@author: 이창현
"""

def solution (n):
    sum = 0
    i = 1
    for i in range(n) :
        if n%i+1 == 0:
            sum += i
    return sum

def sumDivisor(num):
    # num / 2 의 수들만 검사하면 성능 약 2배 향상잼
    return num + sum([i for i in range(1, (num // 2) + 1) if num % i == 0])


def sumDivisor1(num):
    return sum([i for i in range(1,num+1) if num%i==0])