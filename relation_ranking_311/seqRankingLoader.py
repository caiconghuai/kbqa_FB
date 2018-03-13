#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2017-12-12
import sys, os
import pickle
import numpy as np
import random
import torch
from torch.autograd import Variable
sys.path.append('../vocab')
sys.path.append('../tools')
import virtuoso
from args import get_args
args = get_args()

def create_seq_ranking_data(qa_data, word_vocab, rel_sep_vocab, save_path):
    seqs = []
    seq_len = []
    pos_rel1 = []
    pos_rel2 = []
    neg_rel1 = []
    neg_rel2 = []
    batch_index = -1    # the index of sequence batches
    seq_index = 0       # sequence index within each batch
    pad_index = word_vocab.lookup(word_vocab.pad_token)

    data_list = pickle.load(open(qa_data, 'rb'))

    def get_separated_rel_id(relation):
        rel = relation.split('.')
        rel1 = '.'.join(rel[:-1])
        rel2 = rel[-1]
        rel1_id = rel_sep_vocab[0].lookup(rel1)
        rel2_id = rel_sep_vocab[1].lookup(rel2)
        return rel1_id, rel2_id

    for data in data_list:
        tokens = data.question_pattern.split()
        can_rels = []
        if hasattr(data, 'cand_sub') and data.subject in data.cand_sub:
            can_rels = data.cand_rel
        else:
            can_subs = virtuoso.str_query_id(data.text_subject)
            for sub in can_subs:
                can_rels.extend(virtuoso.id_query_out_rel(sub))
            can_rels = list(set(can_rels))

        if seq_index % args.batch_size == 0:
            seq_index = 0
            batch_index += 1
            seqs.append(torch.LongTensor(args.batch_size, len(tokens)).fill_(pad_index))
            seq_len.append(torch.LongTensor(args.batch_size).fill_(1))
            pos_rel1.append(torch.LongTensor(args.batch_size).fill_(pad_index))
            pos_rel2.append(torch.LongTensor(args.batch_size).fill_(pad_index))
            neg_rel1.append(torch.LongTensor(args.neg_size,
                                             args.batch_size).random_(0, len(rel_sep_vocab[0])))
            neg_rel2.append(torch.LongTensor(args.neg_size,
                                             args.batch_size).random_(0, len(rel_sep_vocab[1])))
            print('batch: %d' %batch_index)

        seqs[batch_index][seq_index, 0:len(tokens)] = torch.LongTensor(word_vocab.convert_to_index(tokens))
        seq_len[batch_index][seq_index] = len(tokens)

        pos1, pos2 = get_separated_rel_id(data.relation)
        pos_rel1[batch_index][seq_index] = pos1
        pos_rel2[batch_index][seq_index] = pos2

        for j, neg_rel in enumerate(can_rels):
            if j >= args.neg_size or neg_rel == data.relation:
                continue
            neg1, neg2 = get_separated_rel_id(neg_rel)
            if not neg1 or not neg2:
                continue
            neg_rel1[batch_index][j,seq_index] = neg1
            neg_rel2[batch_index][j,seq_index] = neg2

        seq_index += 1

    torch.save((seqs, seq_len, pos_rel1, pos_rel2, neg_rel1, neg_rel2), save_path)

class SeqRankingSepratedLoader():
    def __init__(self, infile, device=-1):
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

class CandidateRankingLoader():
    def __init__(self, qa_pattern_file, word_vocab, rel_vocab, device=-1):
        self.qa_pattern = pickle.load(open(qa_pattern_file, 'rb'))
        self.batch_num = len(self.qa_pattern)
        self.word_vocab = word_vocab
        self.rel_vocab = rel_vocab
        self.pad_index = word_vocab.lookup(word_vocab.pad_token)
        self.device = device

    def next_question(self):
        for data in self.qa_pattern:
            if not hasattr(data, 'cand_rel'):
                self.batch_num -= 1
                continue
#            print(data.subject, len(can_rels))

            tokens = data.question.split()
            seqs = torch.LongTensor(self.word_vocab.convert_to_index(tokens)).unsqueeze(0)
            seq_len = torch.LongTensor([len(tokens)])

            p_rel = data.relation.split('.')
            p_rel1 = '.'.join(p_rel[:-1])
            p_rel2 = p_rel[-1]
            pos_rel1 = torch.LongTensor([self.rel_vocab[0].lookup(p_rel1)])
            pos_rel2 = torch.LongTensor([self.rel_vocab[1].lookup(p_rel2)])

            n_rel = data.cand_rel
            n_rel1, n_rel2 = [], []
            for r in n_rel:
                r = r.split('.')
                n_rel1.append('.'.join(r[:-1]))
                n_rel2.append(r[-1])
            neg_rel1 = torch.LongTensor(self.rel_vocab[0].convert_to_index(n_rel1)).unsqueeze(1)
            neg_rel2 = torch.LongTensor(self.rel_vocab[1].convert_to_index(n_rel2)).unsqueeze(1)
            if self.device>=0:
                seqs, pos_rel1, pos_rel2, neg_rel1, neg_rel2 = seqs.cuda(self.device), pos_rel1.cuda(self.device), pos_rel2.cuda(self.device), neg_rel1.cuda(self.device), neg_rel2.cuda(self.device)
            yield Variable(seqs), Variable(seq_len), Variable(pos_rel1), Variable(pos_rel2), Variable(neg_rel1), Variable(neg_rel2), data

if __name__ == '__main__':
    word_vocab = torch.load('../vocab/vocab.word&rel.pt')
    rel_sep_vocab = torch.load('../vocab/vocab.rel.sep.pt')

    qa_data_path = '../entity_detection/results-5/QAData.label.%s.pkl'
    if not os.path.exists('data'):
        os.mkdir('data')
    save_path = 'data/%s.relation_ranking.pt'

    for tp in ('valid', 'test', 'train'):
        create_seq_ranking_data(qa_data_path % tp, word_vocab, rel_sep_vocab, save_path % tp)
