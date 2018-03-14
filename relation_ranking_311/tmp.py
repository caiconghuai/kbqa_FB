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
word_vocab = torch.load('../vocab/vocab.word&rel.pt')

loader = SeqRankingLoader('data/valid.relation_ranking.pt', 0)
batch_size = loader.batch_size
for i, batch in enumerate(loader.next_batch(False)):
    if i>=1:break
    print(i)
    seqs, seq_len, pos_rel1, pos_rel2, neg_rel1, neg_rel2, pos_rel, pos_len, neg_rel, neg_len = batch
    seqs_trans = seqs.cpu().data.numpy()
    pos_rel_trans1 = pos_rel1.cpu().data.numpy()
    pos_rel_trans2 = pos_rel2.cpu().data.numpy()
    neg_rel_trans1 = np.transpose(neg_rel1.cpu().data.numpy())

    pos_rel_trans = pos_rel.cpu().data.numpy()
    neg_rel_trans = neg_rel.cpu().data.numpy()
    print(neg_len)

    for j in range(5):
        question = ' '.join(word_vocab.convert_to_word(seqs_trans[j]))
        print(question)
        pos_rel_1 = rel_vocab[0].convert_to_word([pos_rel_trans1[j]])
        pos_rel_2 = rel_vocab[1].convert_to_word([pos_rel_trans2[j]])
        print(pos_rel_1, pos_rel_2)

        pos_rel_ = word_vocab.convert_to_word(pos_rel_trans[j])
        print(pos_rel_, pos_len[j])
        for k in range(5):
            neg_rel_ = word_vocab.convert_to_word(neg_rel_trans[k][j])
            print(neg_rel_)

#        neg_rel_ = ' | '.join(rel_vocab[1].convert_to_word(neg_rel_trans[j]))
#        print(neg_rel_)
#        print(neg_rel_trans[j])
'''

loader = CandidateRankingLoader('../entity_detection/results-5/QAData.label.valid.pkl', word_vocab, rel_vocab, 0)
for i, batch in enumerate(loader.next_question()):
    if i > 10:break
    seqs, seq_len, pos_rel1, pos_rel2, neg_rel1, neg_rel2, pos_rel, pos_len, neg_rel, neg_len, data = batch
    seqs_trans = seqs.cpu().data.numpy()
    pos_rel_trans1 = pos_rel1.cpu().data.numpy()
    pos_rel_trans2 = pos_rel2.cpu().data.numpy()
    neg_rel_trans1 = np.transpose(neg_rel1.cpu().data.numpy())
    question = ' '.join(word_vocab.convert_to_word(seqs_trans[0]))
    print(question)
    pos_rel_1 = rel_vocab[0].convert_to_word(pos_rel_trans1)
    pos_rel_2 = rel_vocab[1].convert_to_word(pos_rel_trans2)
    print(pos_rel_1, pos_rel_2)

    pos_rel_trans = pos_rel.squeeze(0).cpu().data.numpy()
    neg_rel_trans = neg_rel.squeeze(1).cpu().data.numpy()
    pos_rel_ = word_vocab.convert_to_word(pos_rel_trans)
    print(pos_rel_)
    print(neg_rel_trans)
    neg_rel_ = word_vocab.convert_to_word(neg_rel_trans[0])
    print(neg_rel_)
'''
