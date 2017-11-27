#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2017-11-19
import os
import sys
import numpy as np
import torch
import pickle
import multiprocessing as mp

sys.path.append('../tools')
import virtuoso

def add_sub_type2qadata(qa_label_data, pid=0):
    log_file = open('logs/log.%d.txt' % pid, 'w')
    qa_data_new = []
    num_with_cand_sub = 0
    right_cand_sub = 0
    for i, data in enumerate(qa_label_data):
        if hasattr(data, 'cand_sub'):
            num_with_cand_sub += 1
            if data.subject in data.cand_sub:
                right_cand_sub += 1
                for sub in data.cand_sub:
                    type = virtuoso.id_query_type(sub)
                    data.add_sub_types(type)
        qa_data_new.append(data)
        if i % 100 == 0:
            log_file.write('[%d] %d / %d\n' % (pid, i, len(qa_label_data)))
    log_file.write('num with cand_sub: %d\n'% num_with_cand_sub)
    log_file.write('num right cand_sub: %d\n'% right_cand_sub)
    log_file.close()
    pickle.dump(qa_data_new, open('data/temp.%d.pickle' %pid, 'wb'))

if __name__ == '__main__':
    tp = sys.argv[1]
    num_process = int(sys.argv[2])

    qa_label_file = '../entity_detection/results/QAData.label.%s.pkl' %tp
    data_list = pickle.load(open(qa_label_file, 'rb'))
    print(tp, len(data_list))

    # Allocate dataload
    length = len(data_list)
    data_per_p = (length + num_process - 1) // num_process

    # Spawn processes
    processes = [
        mp.Process(
            target = add_sub_type2qadata,
            args = (data_list[pid*data_per_p:(pid+1)*data_per_p],
                    pid)
        )
        for pid in range(num_process)
    ]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    # Merge all data [this will preserve the order]
    new_data_list = []
    for p in range(num_process):
        temp_fn = 'data/temp.%d.pickle'%(p)
        new_data_list.extend(pickle.load(open(temp_fn, 'rb')))

    pickle.dump(new_data_list, open('data/QAData.cand.%s.pkl'%(tp), 'wb'))
 
    # Remove temp data
    for p in range(num_process):
        temp_fn = 'data/temp.%d.pickle'%(p)
        os.remove(temp_fn)
