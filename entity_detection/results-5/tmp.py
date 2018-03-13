#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2018-03-13
import pickle
import sys
sys.path.append('../../tools')

qa_data = pickle.load(open('QAData.label.train.pkl', 'rb'))
new_qa_data = []
for data in qa_data:
    new_qa_data.append((data, len(data.question_pattern.split())))

data_list = [data[0] for data in sorted(new_qa_data, key = lambda data: data[1], reverse=True)]
pickle.dump(data_list, open('QAData.label.train.pkl2', 'wb'))
