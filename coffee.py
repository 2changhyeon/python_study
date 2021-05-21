# -*- coding: utf-8 -*-
"""
Created on Fri May 21 18:02:07 2021

@author: 이창현
"""

coffee = 10
while True:
    money = int(input("돈을 넣어 주세요."))
    if money == 300:
        print("커피를 준다.")
        coffee -= 1
    elif money > 300:
        print("커피를 준다.")
        coffee -= 1
        money -= 300
        print("잔돈반황 : %d" %money)
    else:
        print("잔액이 부족합니다.")
        print("잔여커피량 %d" %coffee)
    if coffee == 0:
        print("커피가 다 떨어졌습니다. 판매를 중지합니다.")
        break
    