#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2017-11-09
import sys
import torch
from seqRankingLoader import *
import numpy as np
sys.path.append('../vocab')

rel_vocab = torch.load('../vocab/vocab.rel.sep.pt')
word_vocab = torch.load('../vocab/vocab.word.pt')
word_vocab.add_start_token()

loader = SeqRankingLoader('data/train.relation_ranking.word.pt', 0)
batch_size = loader.batch_size
for i, batch in enumerate(loader.next_batch(False)):
    if i>1:break
    print(i)
    seqs, seq_len, pos_rel, pos_len, neg_rel, neg_len = batch
    print(seqs.size())
    seqs_trans = seqs.cpu().data.numpy()
    pos_rel_trans = pos_rel.cpu().data.numpy()
    pos_len_trans = pos_len.cpu().numpy()
    print(pos_len_trans)
    neg_rel_trans = neg_rel.cpu().data.numpy()
    neg_len_trans = neg_len.cpu().numpy()
    for j in range(5):
        question = ' '.join(word_vocab.convert_to_word(seqs_trans[j]))
        print(question)
        pos_rel_ = word_vocab.convert_to_word(pos_rel_trans[j])
        print(pos_rel_)
        print(neg_len_trans[:, j])
        for k in range(5):
            neg_rel_ = word_vocab.convert_to_word(neg_rel_trans[k][j])
            print(neg_rel_)
'''

loader = CandidateRankingLoader('../data/QAData.label.pattern.valid.pkl', word_vocab, 0)
for i, batch in enumerate(loader.next_question()):
    if i > 10:break
    seqs, seq_len, pos_rel, pos_len, neg_rel, neg_len, data = batch
    seqs_trans = seqs.cpu().data.numpy()
    pos_rel_trans = pos_rel.cpu().data.numpy()
    neg_rel_trans = neg_rel.squeeze(1).cpu().data.numpy()
    question = ' '.join(word_vocab.convert_to_word(seqs_trans[0]))
    print(question)
    pos_rel_ = word_vocab.convert_to_word(pos_rel_trans[0])
    print(pos_rel_)
    neg_rel_ = word_vocab.convert_to_word(neg_rel_trans[0])
    print(neg_rel_)

'''

