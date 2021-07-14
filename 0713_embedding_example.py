# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 11:19:32 2021

@author: PC
"""

import nltk
#nltk.download('punkt')

import urllib.request
import zipfile
from lxml import etree
import re
from nltk.tokenize import word_tokenize, sent_tokenize

from gensim.models import Word2Vec, KeyedVectors
#rllib.request.urlretrieve("https://raw.githubusercontent.com/GaoleMeng/RNN-and-FFNN-textClassification/master/ted_en-20160408.xml", filename="ted_en-20160408.xml")

targetXML = open('ted_en-20160408.xml','r',encoding = 'UTF8')
target_text = etree.parse(targetXML)

# 필요한 내용만 뽑아온다.
parse_text = '\n'.join(target_text.xpath('//content/text()'))

# 괄호로 구성된 내용 제거
content_text = re.sub(r'\([^)]*\)','',parse_text)

# 정리가 끝난 데이터 nltk를 이용해 문장 tokenize
sent_text = sent_tokenize(content_text)

# 저장할 곳
normalized_text = []

# 구두점 제거, 전부 소문자화하고 저장
for string in sent_text:
    tokens = re.sub(r"[^a-z0-9]+"," ",string.lower())
    normalized_text.append(tokens)

# 각문장에 대해서 단어 tokenize 진행
result = [word_tokenize(sentence) for sentence in normalized_text]

"""
print("{}".format(len(result)))

for line in result[:3]:
    print(line)
    
"""
model = Word2Vec(sentences = result, vector_size=100, window =5 , min_count=5, workers=4, sg = 0)
"""
vector_size = 워드 벡터의 특징값, 임베딩된 벡터의 차원
window = 컨텍스트 윈도우 크기
min_count = 단어의 최소 빈도수 제한
workers = 학습을 위한 프로세스 수
sg = 0 CBOW , 1 Skip-gram
"""

# 입력한 단어에 대해서 가장 유사한 단어를 출력하는 함수
#model_result = model.wv.most_similar("man")
#print(model_result)

#모델 컴퓨터 파일로 저장
model.wv.save_word2vec_format('eng_w2v')
#저장한 모델 불러오기
loaded_model = KeyedVectors.load_word2vec_format("eng_w2v")

model_result = model.wv.most_similar("man")
print(model_result)