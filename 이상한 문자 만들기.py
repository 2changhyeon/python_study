# -*- coding: utf-8 -*-
"""
Created on Sat May 22 17:26:01 2021

@author: 이창현
"""

def solution(s):
    a = ""
    sp = s.split(' ')
    for i in range(len(sp)):
        for j in range(len(sp[i])):
            if j % 2 == 0:
                a+=sp[i][j].upper()
            else:
                a+=sp[i][j].lower()
        if i != len(sp)-1:
            a+=" "
    return a

print(solution("try hello world"))

