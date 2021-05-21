# -*- coding: utf-8 -*-
"""
Created on Fri May 21 20:57:06 2021

@author: 이창현
"""


#ex1

def is_odd(a):
    if a % 2 == 1:
        return True
    else:
        return False

print(is_odd(3))
print(is_odd(4))


odd = lambda a: True if a % 2 == 1 else False

#ex2

def avg_number(*a):
    result = 0
    for i in a:
        result += i
    return result/len(a)
x = avg_number(1,2,3,4)
print(x)

#ex3


input1 =input("1번째 숫자를 입력하세요")
input2 =input("2번째 숫자를 입력하세요")
result = int(input1) + int(input2)
print(result)
