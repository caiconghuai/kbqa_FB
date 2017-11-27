#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2017-11-24
import pickle
import sys
sys.path.append('../tools/')

if __name__ == '__main__':
    tp = sys.argv[1]
    '''
    qa_data_new = []
    qa_data = pickle.load(open('../entity_detection/results/QAData.label.%s.pkl' %tp, 'rb'))
    qa_save_path = open('../entity_detection/results-2/QAData.label.%s.pkl' %tp, 'wb')
    for data in qa_data:
        if hasattr(data, 'cand_rel'):
            data.remove_duplicate()
        qa_data_new.append(data)
    pickle.dump(qa_data_new, qa_save_path)
    '''
    qa_data = pickle.load(open('../entity_detection/results-2/QAData.label.%s.pkl' %tp, 'rb'))
    qa_data_new = pickle.load(open('data/QAData.cand.%s.pkl' %tp, 'rb'))
    save_path = open('data2/QAData.cand.%s.pkl' %tp, 'wb')
    for idx, data in enumerate(qa_data):
        if hasattr(data, 'cand_rel'):
            qa_data_new[idx].cand_rel = data.cand_rel
    pickle.dump(qa_data_new, save_path)
