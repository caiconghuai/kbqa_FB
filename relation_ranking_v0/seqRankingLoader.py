#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2017-11-07
import sys, os
import pickle
import numpy as np
import random
import torch
from torch.autograd import Variable
sys.path.append('../vocab')
sys.path.append('../tools')

def create_seq_ranking_data(batch_size, qa_data, word_vocab, rel_vocab):
    file_type = qa_data.split('.')[-2]
    log_file = open('data/%s.relation_ranking.txt' %file_type, 'w')
    seqs = []
    pos_rel = []
    batch_index = -1    # the index of sequence batches
    seq_index = 0       # sequence index within each batch
    pad_index = word_vocab.lookup(word_vocab.pad_token)

    data_list = pickle.load(open(qa_data, 'rb'))
    for data in data_list:
        tokens = data.question.split()
        if len(tokens) <= 1:    # filter out any question that has only one word
            continue
        log_file.write('%s\t%s\n' %(data.question, data.relation))

        if seq_index % batch_size == 0:
            seq_index = 0
            batch_index += 1
            seqs.append(torch.LongTensor(len(tokens), batch_size).fill_(pad_index))
            pos_rel.append(torch.LongTensor(batch_size).fill_(pad_index))
            print('batch: %d' %batch_index)

        seqs[batch_index][0:len(tokens), seq_index] = torch.LongTensor(word_vocab.convert_to_index(tokens))
        pos_rel[batch_index][seq_index] = rel_vocab.lookup(data.relation)
        seq_index += 1

    torch.save((seqs, pos_rel), 'data/%s.relation_ranking.pt' %file_type)

class SeqRankingLoader():
    def __init__(self, infile, neg_size, neg_range, device=-1):
        self.seqs, self.pos_rel = torch.load(infile)
        self.batch_size = self.seqs[0].size(1)
        self.batch_num = len(self.seqs)

        # for negative sampling
        self.neg_size = neg_size
        self.neg_range = neg_range
        self._pos_rel = torch.LongTensor(1, self.batch_size).expand(self.neg_size, self.batch_size)
        self._neg_rel = torch.LongTensor(self.neg_size, self.batch_size)
        self.neg_rel = self._neg_rel

        if device >=0:
            for i in range(self.batch_num):
                self.seqs[i] = self.seqs[i].cuda(device)
                self.pos_rel[i] = self.pos_rel[i].cuda(device)
            self.neg_rel = self._neg_rel.cuda(device)

    def next_batch(self, shuffle=True):
        if shuffle:
            indices = torch.randperm(self.batch_num)
        else:
            indices = range(self.batch_num)

        for i in indices:
            self._pos_rel.copy_(self.pos_rel[i])
            self._neg_rel.random_(0, self.neg_range)
            while(torch.sum(torch.eq(self._neg_rel, self._pos_rel)) > 0):
                self._neg_rel.masked_fill_(torch.eq(self._neg_rel, self._pos_rel), \
                                           random.randint(0, self.neg_range-1))
            self.neg_rel.copy_(self._neg_rel)
            yield Variable(self.seqs[i]), Variable(self.pos_rel[i]), Variable(self.neg_rel)

if __name__ == '__main__':
    word_vocab = torch.load('../vocab/vocab.word.pt')
    rel_vocab = torch.load('../vocab/vocab.rel.pt')
    create_seq_ranking_data(128, '../data/QAData.test.pkl', word_vocab, rel_vocab)
