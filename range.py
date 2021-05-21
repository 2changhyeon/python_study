# -*- coding: utf-8 -*-
"""
Created on Fri May 21 18:16:24 2021

@author: 이창현
"""
#len, range 활용
marks = [90,25,67,35,80]
for number in range(len(marks)):
    if marks[number] < 60:
        continue
    print("%d번 학생 합격" %(number+1))

# 2중 for 
for i in range(2,10):
    for j in range(1,10):
        print(i*j, end=" ")
        
    print('')
    
#
a = [1,2,3,4]
result = []
for num in a:
    result.append(num*3)

print(result)

result = [num * 3 for num in a if num %2==0]
print(result)

result = [x*y for x in range(2,10)
              for y in range(1,10)]
print(result)


result = 0
i = 1
while i <= 1000:
    if i%3 == 0:
        result += i
    
    i += 1
    
print(result)
