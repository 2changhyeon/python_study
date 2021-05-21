# -*- coding: utf-8 -*-
"""
Created on Fri May 21 21:29:51 2021

@author: 이창현
"""

class FourCal:
    def __init__(self,first,second): #생성자라서 객체생성될때 자동으로 호출된다.
        self.first = first
        self.second = second
    
    def setdata(self, first, second):
        self.first = first
        self.second = second
    
    def add(self):
        result = self.first + self.second
        return result
    
    def sub(self):
        result = self.first - self.second
        return result
    
    def mul(self):
        result = self.first * self.second
        return result
    
    def div(self):
        if self.second == 0:
            return 0
        else:    
            result = self.first / self.second
            return result
    
    

a = FourCal(1,2)
b = FourCal(4,2)

a.setdata(1, 2)

print(a.add(), a.sub(), a.mul(), a.div(),b.add(), b.sub(), b.mul(), b.div())


class MoreFourCal(FourCal):
    def pow(self):
        result = self.first ** self.second
        return result

a = MoreFourCal(4, 2)

print(a.pow())
print(a.add())
print(a.sub())
print(a.mul())
print(a.div())

