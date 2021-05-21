# -*- coding: utf-8 -*-
"""
Created on Fri May 21 18:47:13 2021

@author: 이창현
"""
def say_myself(name, old, man=True):
    print("나의 이름은 %s" %name)
    print("나의 나이는 %d" %old)
    if man:
        print("남자")
    else:
        print("여자")

say_myself("chang", 26, True)


a=1
def vartest(a):
    a = a+1
    return a

vartest(a)

print(a)


a = 1

def vartest2():
    global a
    a = a + 1

vartest2()
print(a)

add = lambda a, b: a+b
result = add(3,4)
print(result)