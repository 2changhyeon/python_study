# -*- coding: utf-8 -*-
"""
Created on Sat May 22 10:07:57 2021

@author: 이창현
"""
arr1 = [[1,2,3],[2,3,4]]
arr2 = [[3,4,5],[5,6,7]]

def solution(arr1, arr2):
    
    answer = [[0 for x in range(len(arr1[0]))] for y in range(len(arr1))]
    print (answer)

    for i in range(len(arr1)):
        for j in range(len(arr1[0])):
            answer[i][j] = arr1[i][j] + arr2[i][j]
    return answer

print(solution(arr1, arr2))
print(range(len(arr1[0])))
print(range(len(arr1)))


def sumMatrix(A,B):
    answer = [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    return answer