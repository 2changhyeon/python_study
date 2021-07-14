# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 11:52:44 2021

@author: PC
"""

import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")

train_data = pd.read_table("ratings.txt")

# NULL 값 존재 유무
print(train_data.isnull().values.any())

#NULL값이 존재하는 행 제거
train_data = train_data.dropna(how = 'any') 

# 제거 확인
print(train_data.isnull().values.any())

# train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

#?

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

okt = Okt()
tokenized_data = []
for sentence in train_data['document']:
    
    #토큰화
    temp_X = okt.morphs(sentence, stem = True)
    # 불용어 제거 
    temp_X = [word for word in temp_X if not word in stopwords] 
    tokenized_data.append(temp_X)

print('리뷰의 최대 길이 :',max(len(l) for l in tokenized_data))
print('리뷰의 평균 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))
plt.hist([len(s) for s in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()