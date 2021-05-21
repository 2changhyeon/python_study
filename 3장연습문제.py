# -*- coding: utf-8 -*-
"""
Created on Fri May 21 18:30:26 2021

@author: 이창현
"""
# ex3
i = 1
while i <= 5:
    print("*" * i)
    i += 1
# ex4    
for i in range (1,101):
    print(i)

# ex5
A = [70,60,55,75,95,90,80,80,85,100]
total = 0
for score in A:
    total += score
    
print("%d" %(total/len(A)))

#ex5
numbers = [1,2,3,4,5]
result=[n*2 for n in numbers if n%2 == 1]
print(result)

#ex6