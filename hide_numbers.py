# -*- coding: utf-8 -*-
"""
Created on Sat May 22 10:59:48 2021

@author: 이창현
"""
a = "01033334444"

def solution0(s):
    return "*"*(len(s)-4) + s[-4:]

def solution1(a):
    b = ''
    for x in range(len(a)-4):
        b += '*'
    b += a[-4:]
    return b
