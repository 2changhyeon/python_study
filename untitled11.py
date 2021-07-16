# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 16:45:50 2021

@author: PC
"""

import pandas as pd
import chardet

recipe = pd.read_csv('train.csv')
recipe.head()
rawdata = open('train.csv', 'rb').read()
result = chardet.detect(rawdata)
charenc = result['encoding']

print(charenc)