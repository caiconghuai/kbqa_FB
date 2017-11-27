#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2017-11-08
import torch
import sys
sys.path.append('../vocab')
from seqMultiLabelLoader import SeqMultiLabelLoader
import numpy as np


word_vocab = torch.load('../vocab/vocab.word.pt')
type_vocab = torch.load('../vocab/vocab.type.pt')
seqs, labels = torch.load('data/train.subject_type.pt')
'''
print(seqs[0])
index_question = np.transpose(seqs[0].numpy())
print(index_question)

for i in range(10):
    question_array = np.array(word_vocab.convert_to_word(index_question[i]))
    print(' '.join(question_array))
'''
loader = SeqMultiLabelLoader('data/test.subject_type.pt', 0)
for i,batch in enumerate(loader.next_batch(False)):
    if i > 1:break
    seq, label = batch
    question = np.transpose(seq.data.cpu().numpy())
    label = label.data.cpu().numpy()
    for i in range(10):
        array = np.array(word_vocab.convert_to_word(question[i]))
        print(' '.join(array))
        print(type_vocab.convert_to_word(np.where(label[i])[0]))
