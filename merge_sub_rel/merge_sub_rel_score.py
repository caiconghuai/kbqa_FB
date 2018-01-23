#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2017-11-20
import os, sys
import pickle
import numpy as np
sys.path.append('../tools/')

def top_sub_rel(data, rel_scores, sub_scores):
    rel_scores = np.array(rel_scores)
    sub_scores = np.array(sub_scores)

    top_sub_ids = np.arange(sub_scores.shape[0])
    top_rel_ids = np.where(rel_scores > 0)[0]
    if len(top_rel_ids) == 0:
        top_rel_ids = np.array([np.argmax(rel_scores)])
    rel_id_dict = {data.cand_rel[rel_id]:i for i, rel_id in enumerate(top_rel_ids)}

    # fill the score matrix
    score_mat = np.zeros((top_sub_ids.shape[0], top_rel_ids.shape[0]))
    for row_idx, sub_id in enumerate(top_sub_ids):
        for rel in data.sub_rels[sub_id]:
            if rel in rel_id_dict:
                col_idx = rel_id_dict[rel]
                score_mat[row_idx, col_idx] = 1

    # compute all the terms
    sub_scores = sub_scores[top_sub_ids]
    rel_scores = rel_scores[top_rel_ids]

#    score_mat = np.outer(sub_scores, rel_scores) * score_mat

    score_mat = np.outer(np.exp(sub_scores), rel_scores) * score_mat

    '''
    alpha = 0.8
    sub_scores = score_mat * sub_scores.reshape(score_mat.shape[0], 1)
    rel_scores = score_mat * rel_scores
    score_mat = (alpha * sub_scores + (1-alpha) * rel_scores)
    '''
    '''
    alpha = 0.55
    score_mat = np.exp(score_mat * alpha + sub_scores.reshape(score_mat.shape[0], 1)*(1-alpha))
    score_mat /= np.sum(score_mat, 0)
    score_mat *= np.exp(rel_scores)
    '''

    top_sub_id, top_rel_id = np.unravel_index(np.argmax(score_mat), score_mat.shape)
    top_sub = data.cand_sub[top_sub_ids[top_sub_id]]
    top_rel = data.cand_rel[top_rel_ids[top_rel_id]]
    print(data.question)
    print(top_sub, data.subject)
    print(top_rel, data.relation)

    return top_sub, top_rel


if __name__ == '__main__':
#    tp = sys.argv[1]
    tp = 'valid'
    data_list = pickle.load(open('data/QAData.cand.%s.pkl' %tp, 'rb'))
    rel_score_list = open('../relation_ranking_v1/results/score-rel-%s.txt' %tp).readlines()
    sub_score_list = open('../subject_type/results/score-sub-%s.txt' %tp).readlines()

    corr_mat = np.zeros((2,2))
    single_corr = 0
    single_num = 0
    idx = 0
    for data in data_list:
#        if idx > 20:break
        if not hasattr(data, 'cand_sub') or data.subject not in data.cand_sub:
            continue
        rel_scores = [float(score) for score in rel_score_list[idx].strip().split(' ')]
        sub_scores = [float(score) for score in sub_score_list[idx].strip().split(' ')]
        top_sub, top_rel = top_sub_rel(data, rel_scores, sub_scores)
        idx += 1
        if len(data.cand_sub) == 1:
            single_num += 1
        if top_sub == data.subject:
            if top_rel == data.relation:
                corr_mat[0,0] += 1
                if len(data.cand_sub) == 1:
                    single_corr += 1
            else:
                corr_mat[0,1] += 1
        else:
            if top_rel == data.relation:
                corr_mat[1,0] += 1
            else:
                corr_mat[1,1] += 1
    print(corr_mat)
    print(single_corr, single_num, single_corr/single_num)
    print(corr_mat[0,0]-single_corr, np.sum(corr_mat)-single_num,
          (corr_mat[0,0]-single_corr)/(np.sum(corr_mat)-single_num))
