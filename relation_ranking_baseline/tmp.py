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

rel_vocab = torch.load('../vocab/vocab.rel.pt')
neg_range = len(rel_vocab)
print(neg_range)
word_vocab = torch.load('../vocab/vocab.word.pt')
'''
loader = SeqRankingLoader('data/train.relation_ranking.pt', neg_range, 0)
batch_size = loader.batch_size
for i, batch in enumerate(loader.next_batch(False)):
    if i>=1:break
    print(i)
    seqs, pos_rel, neg_rel = batch
    seqs_trans = np.transpose(seqs.cpu().data.numpy())
    pos_rel_trans = pos_rel.cpu().data.numpy()
    neg_rel_trans = np.transpose(neg_rel.cpu().data.numpy())
    for j in range(5):
        question = ' '.join(word_vocab.convert_to_word(seqs_trans[j]))
        print(question)
        pos_rel_ = rel_vocab.convert_to_word([pos_rel_trans[j]])
        print(pos_rel_)
        neg_rel_ = ' | '.join(rel_vocab.convert_to_word(neg_rel_trans[j]))
        print(neg_rel_)
        print(neg_rel_trans[j])
'''

loader = CandidateRankingLoader('../entity_detection/results/QAData.label.valid.pkl', word_vocab,
                                rel_vocab, 0)
for i, batch in enumerate(loader.next_question()):
#    if i > 10:break
    seqs, pos_rel, neg_rel = batch
    seqs_trans = np.transpose(seqs.cpu().data.numpy())
    pos_rel_trans = pos_rel.cpu().data.numpy()
    neg_rel_trans = np.transpose(neg_rel.cpu().data.numpy())
#    question = ' '.join(word_vocab.convert_to_word(seqs_trans[0]))
#    print(question)
#    pos_rel_ = rel_vocab.convert_to_word(pos_rel_trans)
#    print(pos_rel_trans, pos_rel_)
#    print(neg_rel_trans)


