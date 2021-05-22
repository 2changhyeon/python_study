# -*- coding: utf-8 -*-
"""
Created on Sat May 22 12:04:09 2021

@author: 이창현
"""

def solution(num):
    count=0
    while num>1:
        if num%2==0:
            num=num//2
            count += 1
        else:
            num=(num*3)+1
            count+=1
    if count>500:return -1

    return count