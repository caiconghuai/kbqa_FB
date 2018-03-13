#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2017-11-08
import torch
import sys
sys.path.append('../vocab')
from seqLabelingLoader import SeqLabelingLoader
import numpy as np

'''
word_vocab = torch.load('../vocab/vocab.word.pt')
seqs, labels = torch.load('data/train.entity_detection.pt')
print(seqs[0])
index_question = np.transpose(seqs[0].numpy())
print(index_question)

for i in range(10):
    question_array = np.array(word_vocab.convert_to_word(index_question[i]))
    print(' '.join(question_array))

loader = SeqLabelingLoader('data/train.entity_detection.pt', 0)
for i,batch in enumerate(loader.next_batch(False)):
    if i > 1:break
    seq, label = batch
    print(seq)
    question = np.transpose(seq.data.cpu().numpy())
    for i in range(10):
        array = np.array(word_vocab.convert_to_word(question[i]))
        print(' '.join(array))
'''
sys.path.append('../tools')
import virtuoso
res = virtuoso.id_query_en_name('fb:m.07b6v8')
print(res)
