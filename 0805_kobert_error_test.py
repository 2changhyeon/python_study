# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 19:03:40 2021

@author: PC
"""
import pandas as pd
import numpy as np
np.random.seed(456)
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
from konlpy.tag import Okt
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score,f1_score
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import BertTokenizer, BertModel, TFBertModel
from transformers import BertConfig, BertForSequenceClassification
from transformers import AdamW
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel



from tokenization_kobert import KoBertTokenizer
from transformers import BertModel, DistilBertModel

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # 텐서플로가 첫 번째 GPU만 사용하도록 제한
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
    print(e)


train=pd.read_csv('../train.csv')
test=pd.read_csv('../test.csv')
sample_submission=pd.read_csv('../sample_submission.csv')


train=train[['과제명', '요약문_연구내용','요약문_한글키워드','label']]
test=test[['과제명', '요약문_연구내용','요약문_한글키워드']]

train['요약문_연구내용'].fillna('NAN', inplace=True)
test['요약문_연구내용'].fillna('NAN', inplace=True)
train['요약문_한글키워드'].fillna('NAN', inplace=True)
test['요약문_한글키워드'].fillna('NAN', inplace=True)

train['data']=train['과제명']+train['요약문_연구내용']+train['요약문_한글키워드']
test['data']=test['과제명']+test['요약문_연구내용']+test['요약문_한글키워드']

# data label
train.index = range(0, len(train))


train['data'] = train['data'].str.replace(r'[-=+,#/\?:^$.@*\"※~>`\'…》\\n\t]+', " ", regex=True)
test['data'] = test['data'].str.replace(r'[-=+,#/\?:^$.@*\"※~>`\'…》]', " ", regex=True)

train['data'] = train['data'].str.replace(r'\t+', " ", regex=True)
test['data'] = test['data'].str.replace(r'\t+', " ", regex=True)

train['data'] = train['data'].str.replace(r'[\\n]+'," ", regex=True)
test['data'] = test['data'].str.replace(r'[\\n]+'," ", regex=True)

train['data'] = train['data'].str.replace(r'[-+]?\d+'," ", regex=True)
test['data'] = test['data'].str.replace(r'[-+]?\d+'," ", regex=True)

train['data'] = train['data'].str.replace("[^가-힣ㄱ-하-ㅣ]", " ", regex=True)
test['data'] = test['data'].str.replace("[^가-힣ㄱ-하-ㅣ]", " ", regex=True)



print(train.head(5))
print(test.head(5))

tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

import collections
def convert_data(data_df):
    global tokenizer
    
    SEQ_LEN = 512 #SEQ_LEN : 버트에 들어갈 인풋의 길이
    
    tokens, masks, segments, targets = [], [], [], []
    
    for i in tqdm(range(len(data_df))):
        # token : 문장을 토큰화함
        # token = tokenizer.encode(data_df[DATA_COLUMN][i], max_length=SEQ_LEN, pad_to_max_length=True)
        token = tokenizer.tokenize(data_df[DATA_COLUMN][i], max_length=SEQ_LEN, pad_to_max_length=True)
        token = tokenizer.convert_tokens_to_ids(token)
       
        # 마스크는 토큰화한 문장에서 패딩이 아닌 부분은 1, 패딩인 부분은 0으로 통일
        num_zeros = token.count(0)
        mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros
        
        # 문장의 전후관계를 구분해주는 세그먼트는 문장이 1개밖에 없으므로 모두 0
        segment = [0]*SEQ_LEN

        # 버트 인풋으로 들어가는 token, mask, segment를 tokens, segments에 각각 저장
        tokens.append(token)
        masks.append(mask)
        segments.append(segment)
        
        # 정답(긍정 : 1 부정 0)을 targets 변수에 저장해 줌
        targets.append(data_df[LABEL_COLUMN][i])

    # tokens, masks, segments, 정답 변수 targets를 numpy array로 지정    
    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    targets = np.array(targets)

    return [tokens, masks, segments], targets

def convert_data2(data_x, data_y):
    global tokenizer
    global max_len
    
    SEQ_LEN = max_len #SEQ_LEN : 버트에 들어갈 인풋의 길이
    
    tokens, masks, segments, targets = [], [], [], []
    
    for i in tqdm(range(len(data_x))):
        # token : 문장을 토큰화함
        # token = tokenizer.encode(data_df[DATA_COLUMN][i], max_length=SEQ_LEN, pad_to_max_length=True)
        token = data_x[i]
       
        # 마스크는 토큰화한 문장에서 패딩이 아닌 부분은 1, 패딩인 부분은 0으로 통일
        
        num_zeros = (token == 0).sum() #token.count(0)
        # print(num_zeros, token)
        mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros
        
        # 문장의 전후관계를 구분해주는 세그먼트는 문장이 1개밖에 없으므로 모두 0
        segment = [0]*SEQ_LEN

        # 버트 인풋으로 들어가는 token, mask, segment를 tokens, segments에 각각 저장
        tokens.append(token)
        masks.append(mask)
        segments.append(segment)
        
        # 정답(긍정 : 1 부정 0)을 targets 변수에 저장해 줌
        targets.append(np.argmax(data_y[i]))

    # tokens, masks, segments, 정답 변수 targets를 numpy array로 지정    
    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    targets = np.array(targets)

    return [tokens, masks, segments], targets

# 위에 정의한 convert_data 함수를 불러오는 함수를 정의
def load_data(pandas_dataframe):
    data_df = pandas_dataframe
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_df[LABEL_COLUMN] = data_df[LABEL_COLUMN].astype(int)
    data_x, data_y = convert_data(data_df)
    return data_x, data_y

  # 위에 정의한 convert_data 함수를 불러오는 함수를 정의
def load_data2(data_x, data_y):
    data_x, data_y = convert_data2(data_x, data_y)
    return data_x, data_y


SEQ_LEN = 512
BATCH_SIZE = 20
# 긍부정 문장을 포함하고 있는 칼럼
DATA_COLUMN = "data"
# 긍정인지 부정인지를 (1=긍정,0=부정) 포함하고 있는 칼럼
LABEL_COLUMN = "label"

# train 데이터를 버트 인풋에 맞게 변환
# Failed to convert a NumPy array to a Tensor (Unsupported object type list).

train_y1, train_x1 = load_data(train)

train_x = tf.ragged.constant(train_x1)
train_y = tf.ragged.constant(train_y1)

# 'tuple' object has no attribute 'astype'

# ValueError: could not broadcast input array from shape (174304,512) into shape (174304
#train_x = tf.convert_to_tensor(train_x1, dtype=tf.float32)
#train_y = tf.convert_to_tensor(train_y1, dtype=tf.float32)
print("...................................................")
print(train_x.dtype)
print(train_y.dtype)

def create_sentiment_bert():
  # 버트 pretrained 모델 로드
  model = TFBertModel.from_pretrained("monologg/kobert", from_pt=True)
  # 토큰 인풋, 마스크 인풋, 세그먼트 인풋 정의
  token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids')
  mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')
  segment_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_segment')
  # 인풋이 [토큰, 마스크, 세그먼트]인 모델 정의
  bert_outputs = model([token_inputs, mask_inputs, segment_inputs])
  dnn_units = 256 #256
  DROPOUT_RATE = 0.2

  bert_outputs = bert_outputs[1]
  # sentiment_first = tf.keras.layers.Dense(3, activation='softmax', kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02))(bert_outputs)
  mid_layer = tf.keras.layers.Dense(dnn_units, activation='relu', kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02))(bert_outputs)
  mid_layer2 = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(mid_layer)
  sentiment_first = tf.keras.layers.Dense(46, activation='softmax', kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02))(mid_layer2)
  sentiment_model = tf.keras.Model([token_inputs, mask_inputs, segment_inputs], sentiment_first)
  # 옵티마이저는 간단하게 Adam 옵티마이저 활용
  sentiment_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
      metrics=['sparse_categorical_accuracy'])
  return sentiment_model

print(train_x.dtype)
print(train_y.dtype)

num_epochs = 1
batch_size = 6

sentiment_model = create_sentiment_bert()
sentiment_model.fit(train_x, train_y, epochs=num_epochs, shuffle=False, batch_size=batch_size)
sentiment_model.save_weights(os.path.join("sentiment_model.h5"))

