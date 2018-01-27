#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2018-01-23

import pickle
import numpy as np

tp = 'test'
v2_file = '../../relation_ranking_v2/results/score-rel-%s.pkl' %tp
v3_file = '../results/score-rel-%s.pkl' %tp

v2_score = pickle.load(open(v2_file, 'rb'))
v3_score = pickle.load(open(v3_file, 'rb'))

def cal_merged_score(alpha):
    n_rel_correct = 0
    
    for score2, s3 in zip(v2_score, v3_score):
        neg_rel, relation, score3 = s3
        neg_score = score2 + score3 * alpha
    
        pred_rel_scores = sorted(zip(neg_rel, neg_score), key=lambda i:i[1], reverse=True)
        pred_rel = pred_rel_scores[0][0]
    
        if pred_rel == relation:
            n_rel_correct += 1

    total = len(v3_score)
    rel_acc = 100 * n_rel_correct / total
    print("%s\taccuracy: %8.6f\tcorrect: %d\ttotal: %d" %(tp, rel_acc, n_rel_correct, total))
    print("-" * 80)

for alpha in np.arange(0.5, 5, 0.1):
    print(alpha)
    cal_merged_score(alpha)
