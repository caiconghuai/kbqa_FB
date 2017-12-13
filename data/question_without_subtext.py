#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2017-12-12
# 把question去掉sub_text，变为pattern  写入QAData.pattern.*.pkl
import sys, os
import pickle
import torch
import numpy as np
sys.path.append('../vocab')
sys.path.append('../tools')

def remove_sub_text_question(qa_data, word_vocab):
    file_type = qa_data.split('.')[-2]
    data_list = pickle.load(open(qa_data, 'rb'))
    new_data_list = []
    for data in data_list:
        if not data.text_attention_indices:
            continue
        tokens = data.question.split()
        # 把sub_text替换成一个特殊字符
        text_indeces = data.text_attention_indices
        tokens[text_indeces[0]] = word_vocab.start_token
        for i in range(len(text_indeces)-1):
            tokens.pop(text_indeces[1])
        data.question = ' '.join(tokens)
        new_data_list.append((data, len(tokens)))

    _data_list = [data[0] for data in sorted(new_data_list, key = lambda data: data[1],
                                             reverse=True)]
    pickle.dump(_data_list, open('QAData.pattern.%s.pkl' %file_type, 'wb'))

if __name__ == '__main__':
    word_vocab = torch.load('../vocab/vocab.word.pt')
    word_vocab.add_start_token()
    remove_sub_text_question('QAData.valid.pkl', word_vocab)
    remove_sub_text_question('QAData.test.pkl', word_vocab)
    remove_sub_text_question('QAData.train.pkl', word_vocab)

