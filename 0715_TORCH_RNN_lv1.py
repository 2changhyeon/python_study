# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:47:50 2021

@author: PC
"""


from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import torch.nn as nn
import torch
import random

def findFiles(path): return glob.glob(path)

#print(findFiles('data/names/*.txt')) # data/names의 모든 문자열 txt파일
# ?는 한자리의 문자를 의미함.

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) # NFD가 뭐지.
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

#print(unicodeToAscii('Ślusàrski'))

category_lines = {}
all_categories = []

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    # 문자열 끝에 .strip() 함수를 사용하면 문자열의 맨앞 맨뒤의 whitespace가 제거됨
    return [unicodeToAscii(line) for line in lines]
    # 읽어온 라인에서 ascii로 바꾸고 리턴
    
for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    # 경로명 path를 root + ext == path가 되도록 쌍 (root, ext)로 분할하는데, 
    # ext는 비어 있거나 마침표로 시작하고 최대 하나의 마침표를 포함합니다. 
    # 기본 이름(basename)의 선행 마침표는 무시됩니다; splitext('.cshrc')는 ('.cshrc', '')를 반환합니다.
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines
    
n_categories = len(all_categories)

#print(category_lines['Italian'][:5])


# all_letters 로 문자의 주소 찾기
def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    # 한개의 문자를 <1x n_letters> Tensor로 변환
    tensor[0][letterToIndex(letter)] = 1
    # letter의 주소에 1을 넣어서 원핫벡터로 만듦
    return tensor
# 이름을 <line_length x 1 x n_letters>
# or 원핫 문자 벡터의 array로 변경
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

#print(letterToTensor('J'))
#print(lineToTensor('Jones').size())



# 네트워크 생성
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN,self).__init__()
        
        self.hidden_size = hidden_size
       
        # input to hidden
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # input to output
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden):
        # cat함수는 concatenate 해주는 함수, concatenate하고자 하는 차원을 증가시킨다.
        # 차원의 수는  유지된다. NxK cat NxK -> Nx2K
        cobined = torch.cat((input, hidden), 1)
        hidden = self.i2h(cobined)
        output = self.i2o(cobined)
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

# letterToTensor 대신에 lineToTensor를 사용한다. 매단계마다 새로운 텐서를 만들지 않기위해서
#input = letterToTensor('A')
input = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output) # 1x n_categories로 나옴


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)


#학습준비 
def categoryFromOutput(output):
    top_n, top_i = output.topk(1) # 텐서의 가장 큰 값 및 주소
    category_i = top_i[0].item()     # 텐서에서 정수 값으로 변경
    return all_categories[category_i], category_i

#print(categoryFromOutput(output))

# 손실함수 RNN 마지막 계층이 LogSoftMax이기때문에, NLLLoss를 사용한다.
# nn.NLLLoss는 nn.LogSoftmax의 log 결과값에 대한 교차 엔트로피 손실 연산(Cross Entropy Loss|Error)입니다.
criterion = nn.NLLLoss()
"""
 학습 루프:
   1. 입력과 목표 Tensor 생성
   2. 0 로 초기화된 은닉 상태 생성
   3. 각 문자를 읽기
        - 다음 문자를 위한 은닉 상태 유지
   4. 목표와 최종 출력 비교
   5. 역전파
   6. 출력과 손실 반환 
"""

learning_rate = 0.005
 # lr을 너무 높게 설정하면 발산할 수 있고, 너무 낮으면 학습이 되지 않을 수 있습니다.

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # 매개변수의 경사도에 학습률을 곱해서 그 매개변수의 값에 더합니다.
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()



import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000



# 도식화를 위한 손실 추적
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # iter 숫자, 손실, 이름, 추측 화면 출력
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # 현재 평균 손실을 전체 손실 리스트에 추가
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


