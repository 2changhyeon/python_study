# -*- coding: utf-8 -*-
"""
Created on Fri May 21 21:59:13 2021

@author: 이창현
"""
#클래스 변수

class Family:
    lastname="김"
    
print(Family.lastname)
a = Family()
b = Family()
print(a.lastname)
print(b.lastname)

Family.lastname = "박"

print(a.lastname)


PI = 3.141592
class Math:
    def solv(self,r):
        return PI * (r ** 2)
    def add (a,b):
        return a+b
