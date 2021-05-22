# -*- coding: utf-8 -*-
"""
Created on Sat May 22 16:50:36 2021

@author: 이창현
"""


def solution(n):
    sqrt = n ** (1/2)
    if sqrt % 1 == 0:
        return (sqrt + 1) ** 2
    return -1
