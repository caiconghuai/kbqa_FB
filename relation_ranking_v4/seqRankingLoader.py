#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2018-01-27

import sys, os
import pickle
import numpy as np
import random
import torch
from torch.autograd import Variable
from args import get_args
sys.path.append('../vocab')
sys.path.append('../tools')

args = get_args()

# 将rel分成word level; 直接对create_seq_ranking_data生成的pt文件进行处理
def create_seq_ranking_data_word(batch_size, batch_data, rel_vocab, word_vocab):
    file_type = batch_data.split('.')[0]
    pos_rel_list = []
    neg_rel_list = []
    seqs_list = []
    seq_len_list = []
    pos_rel_len = []
    neg_rel_len = []
    batch_index = 0
    pad_index = word_vocab.lookup(word_vocab.pad_token)

    rel_max_len = args.rel_maxlen

    _seqs, _pos_rel, _neg_rel, _neg_rel_size = torch.load(batch_data)
    for seqs, pos, neg_array, n_size in zip(_seqs, _pos_rel, _neg_rel, _neg_rel_size):
        print('batch: %d' %batch_index)

        pos_rel = torch.LongTensor(batch_size, rel_max_len).fill_(pad_index)
        pos_id = pos.numpy()
        pos = rel_vocab.convert_to_word(pos_id)
        pos_len = torch.Tensor(batch_size).fill_(1)

        neg_size = max(n_size)
        neg_rel = torch.LongTensor(neg_size, batch_size, rel_max_len).fill_(pad_index)
        neg_len = torch.Tensor(neg_size, batch_size).fill_(1)

        seqs = torch.transpose(seqs, 0, 1)
        seq_len = torch.Tensor(batch_size).fill_(1)

        for idx in range(batch_size):
            if pos[idx] == rel_vocab.pad_token:continue

            seq_len[idx] = len(seqs[idx])
            for w in seqs[idx]:
                if w == pad_index:
                    seq_len[idx] -= 1

            pos_words = []
            p = pos[idx][3:].split('.')
            for i in p:
                pos_words.extend(i.split('_'))
            if len(pos_words)>rel_max_len:
                print(len(pos_words))
            pos_rel[idx, 0:len(pos_words)] = torch.LongTensor(word_vocab.convert_to_index(pos_words))
            pos_len[idx] = len(pos_words)

            can_rels = neg_array[idx]
            if pos_id[idx] in can_rels:
                can_rels.remove(pos_id[idx])
            can_len = len(can_rels)
            for i in range(neg_size-can_len):
                tmp = random.randint(2, len(rel_vocab)-1)
                while(tmp == pos_id[idx]):
                    tmp = random.randint(2, len(rel_vocab)-1)
                can_rels.append(tmp)
            can_rels = rel_vocab.convert_to_word(can_rels)
            for j, neg in enumerate(can_rels):
                neg_words = []
                neg = neg[3:].split('.')
                for i in neg:
                    neg_words.extend(i.split('_'))
                if len(neg_words)>rel_max_len:
                    print(len(neg_words))
                neg_rel[j, idx, 0:len(neg_words)] = torch.LongTensor(word_vocab.convert_to_index(neg_words))
                neg_len[j, idx] = len(neg_words)

        seqs_list.append(seqs)
        seq_len_list.append(seq_len)
        pos_rel_list.append(pos_rel)
        pos_rel_len.append(pos_len)
        neg_rel_list.append(neg_rel)
        neg_rel_len.append(neg_len)

        batch_index += 1
    torch.save((seqs_list, seq_len_list, pos_rel_list, pos_rel_len, neg_rel_list, neg_rel_len),
              '%s.relation_ranking.word.pt' %file_type)

class SeqRankingLoader():
    def __init__(self, infile, device=-1):
        self.seqs, self.seq_len, self.pos_rel, self.pos_len, self.neg_rel, self.neg_len = torch.load(infile)
        self.batch_size = self.seqs[0].size(0)
        self.batch_num = len(self.seqs)

        if device >= 0:
            for i in range(self.batch_num):
                self.seqs[i] = self.seqs[i].cuda(device)
#                self.seq_len[i] = self.seq_len[i].cuda(device)
                self.pos_rel[i] = self.pos_rel[i].cuda(device)
#                self.pos_len[i] = self.pos_len[i].cuda(device)
                self.neg_rel[i] = self.neg_rel[i].cuda(device)
#                self.neg_len[i] = self.neg_len[i].cuda(device)

    def next_batch(self, shuffle=True):
        if shuffle:
            indices = torch.randperm(self.batch_num)
        else:
            indices = range(self.batch_num)
        for i in indices:
            yield Variable(self.seqs[i]), self.seq_len[i], Variable(self.pos_rel[i]), self.pos_len[i], Variable(self.neg_rel[i]), self.neg_len[i]

class CandidateRankingLoader():
    def __init__(self, qa_pattern_file, word_vocab, device=-1):
        self.qa_pattern = pickle.load(open(qa_pattern_file, 'rb'))
        self.batch_num = len(self.qa_pattern)
        self.word_vocab = word_vocab
        self.pad_index = word_vocab.lookup(word_vocab.pad_token)
        self.device = device

    def next_question(self):
        for data in self.qa_pattern:
            if not hasattr(data, 'cand_rel'):
                self.batch_num -= 1
                continue

            tokens = data.question.split()
            seqs = torch.LongTensor(self.word_vocab.convert_to_index(tokens)).unsqueeze(0)
            seq_len = torch.LongTensor([len(tokens)])

            pos_words = []
            p = data.relation[3:].split('.')
            for i in p:
                pos_words.extend(i.split('_'))
            pos_rel = torch.LongTensor(args.rel_maxlen).fill_(self.pad_index)
            pos_rel[0:len(pos_words)] = torch.LongTensor(self.word_vocab.convert_to_index(pos_words))
            pos_rel = pos_rel.unsqueeze(0)
            pos_len = torch.LongTensor([len(pos_words)])

            neg_rel = torch.LongTensor(len(data.cand_rel), args.rel_maxlen).fill_(self.pad_index)
            neg_len = torch.LongTensor(len(data.cand_rel))
            for idx, rel in enumerate(data.cand_rel):
                neg_words = []
                p = rel[3:].split('.')
                for i in p:
                    neg_words.extend(i.split('_'))
                neg_rel[idx, 0:len(neg_words)] = torch.LongTensor(self.word_vocab.convert_to_index(neg_words))
                neg_len[idx] = len(neg_words)
            neg_rel = neg_rel.unsqueeze(1)

            if self.device>=0:
                seqs, pos_rel, neg_rel = seqs.cuda(self.device), pos_rel.cuda(self.device), neg_rel.cuda(self.device)
            yield Variable(seqs), seq_len, Variable(pos_rel), pos_len, Variable(neg_rel), neg_len, data



if __name__ == '__main__':
    word_vocab = torch.load(args.vocab_file)
    rel_vocab = torch.load(args.rel_vocab_file)
    create_seq_ranking_data_word(64, 'data/valid.relation_ranking.pt', rel_vocab, word_vocab)
    create_seq_ranking_data_word(64, 'data/test.relation_ranking.pt', rel_vocab, word_vocab)
    create_seq_ranking_data_word(64, 'data/train.relation_ranking.pt', rel_vocab, word_vocab)
