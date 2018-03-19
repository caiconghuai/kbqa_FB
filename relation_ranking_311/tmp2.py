#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2018-03-15
import pickle
import sys
sys.path.append('../tools')
qa_data = '../entity_detection/results-5/QAData.label.test.pkl'
data_list = pickle.load(open(qa_data, 'rb'))

sums = 0
cnt = 0
cnt50 = 0
cnt90 = 0
for data in data_list:
    if hasattr(data, 'cand_sub'):
        cnt += 1
        sums += len(data.cand_rel)
        if len(data.cand_rel) > 50:
            cnt50 += 1
        if len(data.cand_rel) > 100:
            cnt90 += 1
print(sums, cnt, sums/cnt)
print(cnt50, cnt50/cnt)
print(cnt90, cnt90/cnt)
