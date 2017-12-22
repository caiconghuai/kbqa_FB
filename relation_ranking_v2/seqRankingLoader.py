#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2017-12-12
# 负样本不是随机采样，而是取与data.subject相连的relation
# 用去掉sub_text后的question训练
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
#    log_file = open('data/%s.relation_ranking.txt' %file_type, 'w')
    seqs = []
    pos_rel = []
    neg_rel = []
    neg_rel_size = []
    batch_index = -1    # the index of sequence batches
    seq_index = 0       # sequence index within each batch
    pad_index = word_vocab.lookup(word_vocab.pad_token)

    data_list = pickle.load(open(qa_data, 'rb'))
    for data in data_list:
        tokens = data.question.split()
        #取相同name的所有subject相连的rel作为负样本
        can_subs = virtuoso.str_query_id(data.text_subject)
        can_rels = []
        for sub in can_subs:
            can_rels.extend(virtuoso.id_query_out_rel(sub))
        can_rels = list(set(can_rels))  # 去除重复的rel
#        log_file.write('%s\t%s\t%s\n' %(data.question, data.relation, can_rels))

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

# 将rel分成两部分，分别预测; 直接对create_seq_ranking_data生成的pt文件进行处理
def create_seq_ranking_data_separated(batch_size, batch_data, rel_vocab, rel_sep_vocab):
    file_type = batch_data.split('.')[0]
    pos_rel1 = []
    pos_rel2 = []
    neg_rel1 = []
    neg_rel2 = []
    seqs = []
    seq_len = []
    batch_index = 0    # the index of sequence batches
    pad_index = rel_vocab.lookup(rel_vocab.pad_token)

    s, p, n, neg_rel_size = torch.load(batch_data)
    for seq, pos_rel, neg_rel_array, _neg_size in zip(s, p, n, neg_rel_size):
        print('batch: %d' %batch_index)
        _pos_rel1 = torch.LongTensor(batch_size).fill_(pad_index)
        _pos_rel2 = torch.LongTensor(batch_size).fill_(pad_index)
        pos_rel = rel_vocab.convert_to_word(pos_rel.numpy())

        neg_size = max(_neg_size)
        _neg_rel1 = torch.LongTensor(neg_size, batch_size)
        _neg_rel1.random_(0, len(rel_sep_vocab[0]))
        _neg_rel2 = torch.LongTensor(neg_size, batch_size)
        _neg_rel2.random_(0, len(rel_sep_vocab[1]))

        _seq = torch.transpose(seq, 0, 1)
        _seq_len = torch.LongTensor(batch_size).fill_(1)

        for i, pos in enumerate(pos_rel):
            if pos == rel_vocab.pad_token:continue

            _seq_len[i] = len(_seq[i])
            for w in _seq[i]:
                if w == pad_index:
                    _seq_len[i] -= 1

            _pos = pos.split('.')
            p_rel1 = '.'.join(_pos[:-1])
            p_rel2 = _pos[-1]
            _pos_rel1[i] = rel_sep_vocab[0].lookup(p_rel1)
            _pos_rel2[i] = rel_sep_vocab[1].lookup(p_rel2)

            can_rels = neg_rel_array[i]
            can_rels = rel_vocab.convert_to_word(can_rels)
            for j, neg in enumerate(can_rels):
                if neg == pos:continue
                if neg == rel_vocab.unk_token:continue #从处理的pt转回来时有些是unk。。待改
                neg = neg.split('.')
                n_rel1 = '.'.join(neg[:-1])
                n_rel2 = neg[-1]
                _neg_rel1[j,i] = rel_sep_vocab[0].lookup(n_rel1)
                _neg_rel2[j,i] = rel_sep_vocab[1].lookup(n_rel2)

        pos_rel1.append(_pos_rel1)
        pos_rel2.append(_pos_rel2)
        neg_rel1.append(_neg_rel1)
        neg_rel2.append(_neg_rel2)
        seqs.append(_seq)
        seq_len.append(_seq_len)
        batch_index += 1

    torch.save((seqs, seq_len, pos_rel1, pos_rel2, neg_rel1, neg_rel2),
               '%s.relation_ranking.separated2.pt' %file_type)

class SeqRankingSepratedLoader():
    def __init__(self, infile, neg_range, device=-1):
        self.seqs, self.seq_len, self.pos_rel1, self.pos_rel2, self.neg_rel1, self.neg_rel2 = torch.load(infile)
        self.batch_size = self.seqs[0].size(0)
        self.batch_num = len(self.seqs)

        if device >=0:
            for i in range(self.batch_num):
                self.seqs[i] = self.seqs[i].cuda(device)
                self.seq_len[i] = self.seq_len[i].cuda(device)
                self.pos_rel1[i] = self.pos_rel1[i].cuda(device)
                self.pos_rel2[i] = self.pos_rel2[i].cuda(device)
                self.neg_rel1[i] = self.neg_rel1[i].cuda(device)
                self.neg_rel2[i] = self.neg_rel2[i].cuda(device)

    def next_batch(self, shuffle = True):
        if shuffle:
            indices = torch.randperm(self.batch_num)
        else:
            indices = range(self.batch_num)
        for i in indices:
            yield Variable(self.seqs[i]), Variable(self.seq_len[i]), Variable(self.pos_rel1[i]), Variable(self.pos_rel2[i]), Variable(self.neg_rel1[i]), Variable(self.neg_rel2[i])

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
    word_vocab.add_start_token()
    '''
    create_seq_ranking_data(64, '../data/QAData.pattern.valid.pkl', word_vocab, rel_vocab)
    create_seq_ranking_data(64, '../data/QAData.pattern.test.pkl', word_vocab, rel_vocab)
    create_seq_ranking_data(64, '../data/QAData.pattern.train.pkl', word_vocab, rel_vocab)
    '''
    rel_sep_vocab = torch.load('../vocab/vocab.rel.sep.pt')
    create_seq_ranking_data_separated(64, 'data/valid.relation_ranking.pt', rel_vocab, rel_sep_vocab)
    create_seq_ranking_data_separated(64, 'data/test.relation_ranking.pt', rel_vocab, rel_sep_vocab)
    create_seq_ranking_data_separated(64, 'data/train.relation_ranking.pt', rel_vocab, rel_sep_vocab)
