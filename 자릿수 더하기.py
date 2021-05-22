# -*- coding: utf-8 -*-
"""
Created on Sat May 22 17:20:33 2021

@author: 이창현
"""
def solution(number):
    return sum([int(i) for i in str(number)])



def solution1(n):
    answer = 0
    result = list(str(n))
    for i in range(len(result)):
        answer += int(result[i])

    return answer