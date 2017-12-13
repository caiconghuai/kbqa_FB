#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2017-11-13
# 负样本不是随机采样，而是取与data.subject相连的relation
# 相当于假设subject全部预测正确。作为baseline
import sys, os
import pickle
import numpy as np
import random
import torch
from torch.autograd import Variable
sys.path.append('../vocab')
sys.path.append('../tools')
import virtuoso

def create_seq_ranking_data(batch_size, qa_data, word_vocab, rel_vocab):
    file_type = qa_data.split('.')[-2]
    log_file = open('data/%s.relation_ranking.txt' %file_type, 'w')
    seqs = []
    pos_rel = []
    neg_rel = []
    neg_rel_size = []
    batch_index = -1    # the index of sequence batches
    seq_index = 0       # sequence index within each batch
    pad_index = word_vocab.lookup(word_vocab.pad_token)

    data_list = pickle.load(open(qa_data, 'rb'))
    for data in data_list:
        max_neg_size = 0
        tokens = data.question.split()
        sub = data.subject
        can_rels = virtuoso.id_query_out_rel(sub)
        if len(tokens) <= 1:    # filter out any question that has only one word
            continue
        log_file.write('%s\t%s\t%s\n' %(data.question, data.relation, can_rels))

        if seq_index % batch_size == 0:
            seq_index = 0
            batch_index += 1
            seqs.append(torch.LongTensor(len(tokens), batch_size).fill_(pad_index))
            pos_rel.append(torch.LongTensor(batch_size).fill_(pad_index))
            neg_rel.append([])
            neg_rel_size.append([])
            print('batch: %d' %batch_index)

        seqs[batch_index][0:len(tokens), seq_index] = torch.LongTensor(word_vocab.convert_to_index(tokens))
        pos_rel[batch_index][seq_index] = rel_vocab.lookup(data.relation)
        neg_rel[batch_index].append(rel_vocab.convert_to_index(can_rels))
        neg_rel_size[batch_index].append(len(can_rels))
        seq_index += 1

    torch.save((seqs, pos_rel, neg_rel, neg_rel_size), 'data/%s.relation_ranking.pt' %file_type)

class SeqRankingLoader():
    def __init__(self, infile, neg_range, device=-1):
        self.seqs, self.pos_rel, self.neg_rel_array, self.neg_sizes = torch.load(infile)
        self.batch_size = self.seqs[0].size(1)
        self.batch_num = len(self.seqs)
        self.neg_range = neg_range
        self.device = device

        if device >=0:
            for i in range(self.batch_num):
                self.seqs[i] = self.seqs[i].cuda(device)
                self.pos_rel[i] = self.pos_rel[i].cuda(device)

    def next_batch(self, shuffle=True):
        if shuffle:
            indices = torch.randperm(self.batch_num)
        else:
            indices = range(self.batch_num)

        for i in indices:
            neg_size = max(self.neg_sizes[i])
            if neg_size > 512:
                print(neg_size)
            neg_rel = torch.LongTensor(neg_size, self.batch_size)
            neg_rel.random_(0, self.neg_range)
            for j, can_rels in enumerate(self.neg_rel_array[i]):
                if len(can_rels)>0:
                    neg_rel[0:len(can_rels), j] = torch.LongTensor(can_rels)
            _pos_rel = self.pos_rel[i].unsqueeze(0).expand(neg_size, self.batch_size).cpu()
            while(torch.sum(torch.eq(neg_rel, _pos_rel)) > 0):
                neg_rel.masked_fill_(torch.eq(neg_rel, _pos_rel), random.randint(0, self.neg_range-1)) # randint的范围包括后面。。。。
            if self.device >= 0:
                neg_rel = neg_rel.cuda(self.device)
            yield Variable(self.seqs[i]), Variable(self.pos_rel[i]), Variable(neg_rel)

class CandidateRankingLoader():
    def __init__(self, qa_data_file, word_vocab, rel_vocab, device=-1):
        self.qa_data = pickle.load(open(qa_data_file, 'rb'))
        self.batch_num = len(self.qa_data)
        self.word_vocab = word_vocab
        self.rel_vocab = rel_vocab
        self.pad_index = word_vocab.lookup(word_vocab.pad_token)
        self.device = device

    def next_question(self):
        for data in self.qa_data:
            tokens = data.question.split()
            '''
            if len(tokens) <= 1:    # filter out any question that has only one word
                self.batch_num -= 1
                continue
            '''
            if not hasattr(data, 'cand_rel'):
                self.batch_num -= 1
                continue
#            print(data.subject, len(can_rels))

            seqs = torch.LongTensor(self.word_vocab.convert_to_index(tokens)).unsqueeze(1)
            pos_rel = torch.LongTensor([self.rel_vocab.lookup(data.relation)])
            neg_rel = torch.LongTensor(self.rel_vocab.convert_to_index(data.cand_rel)).unsqueeze(1)
            if self.device>=0:
                seqs, pos_rel, neg_rel = seqs.cuda(self.device), pos_rel.cuda(self.device), neg_rel.cuda(self.device)
            yield Variable(seqs), Variable(pos_rel), Variable(neg_rel), data

if __name__ == '__main__':
    word_vocab = torch.load('../vocab/vocab.word.pt')
    rel_vocab = torch.load('../vocab/vocab.rel.pt')
    create_seq_ranking_data(64, '../data/QAData.test.pkl', word_vocab, rel_vocab)
