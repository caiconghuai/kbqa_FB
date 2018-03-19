#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2018-03-19
import sys, os
import pickle
import numpy as np
import torch
from torch.autograd import Variable
sys.path.append('../vocab')
sys.path.append('../tools')
from qa_data import QAData
import virtuoso
from args import get_args
args = get_args()

class CandidateRankingLoader():
    def __init__(self, file_name, word_vocab, rel_sep_vocab, device=-1):
        self.word_vocab = word_vocab
        self.rel_sep_vocab = rel_sep_vocab
        self.pad_index = word_vocab.lookup(word_vocab.pad_token)
        self.device = device
        self.rel_dict = self.create_seq_ranking_data()
        self.data_list = open(file_name)
        self.batch_num = 1

    def create_seq_ranking_data(self):
        vocab_dict = {}
        index = 1
        with open('../others/relation.2M.list') as f:
            for line in f:
                if not line:continue
                line = line.strip()
                if not line:break
                rel_word = line[1:].split('/')
                rel = 'fb:' + '.'.join(rel_word)
                rel1 = 'fb:' + '.'.join(rel_word[:-1])
                rel2 = rel_word[-1]
                id1 = self.rel_sep_vocab[0].lookup(rel1)
                id2 = self.rel_sep_vocab[1].lookup(rel2)
                words = []
                for i in rel_word:
                    words.extend(i.split('_'))
                id = self.word_vocab.convert_to_index(words)
                if not id or not id1 or not id2:
                    print(rel)
                vocab_dict[index] = (id, id1, id2)
                index += 1
        return vocab_dict

    def quesiton2token(self, question):
        tokens = question.split()
        for i, token in enumerate(tokens):
            if token.startswith('#') and token.endswith('#'):
                tokens[i] = 'X'
            elif token == '-lrb-':
                tokens[i] = '('
            elif token == '-rrb-':
                tokens[i] = ')'
        return tokens

    def next_question(self):
        for line in self.data_list:
            if not line:continue
            line = line.strip()
            if not line:break
            pos_index, neg_index, question = line.split('\t')

            tokens = self.quesiton2token(question)
            seqs = torch.LongTensor(self.word_vocab.convert_to_index(tokens)).unsqueeze(0)
            seq_len = torch.LongTensor([len(tokens)])

            pos, pos1, pos2 = self.rel_dict[int(pos_index)]
            pos_rel1 = torch.LongTensor([pos1])
            pos_rel2 = torch.LongTensor([pos2])
            pos_rel = torch.LongTensor(args.rel_maxlen).fill_(self.pad_index)
            pos_rel[0:len(pos)] = torch.LongTensor(pos)
            pos_rel = pos_rel.unsqueeze(0)
            pos_len = torch.LongTensor([len(pos)])

            cand_rel = neg_index.split()
            neg_rel1 = torch.LongTensor(len(cand_rel))
            neg_rel2 = torch.LongTensor(len(cand_rel))
            neg_rel = torch.LongTensor(len(cand_rel), args.rel_maxlen).fill_(self.pad_index)
            neg_len = torch.LongTensor(len(cand_rel))
            for idx, rel in enumerate(cand_rel):
                neg, neg1, neg2 = self.rel_dict[int(rel)]
                neg_rel1[idx] = neg1
                neg_rel2[idx] = neg2
                neg_rel[idx, 0:len(neg)] = torch.LongTensor(neg)
                neg_len[idx] = len(neg)
            neg_rel1.unsqueeze_(1)
            neg_rel2.unsqueeze_(1)
            neg_rel.unsqueeze_(1)

            if self.device>=0:
                seqs, pos_rel1, pos_rel2, neg_rel1, neg_rel2, pos_rel, neg_rel = \
                seqs.cuda(self.device), pos_rel1.cuda(self.device), pos_rel2.cuda(self.device), \
                neg_rel1.cuda(self.device), neg_rel2.cuda(self.device), pos_rel.cuda(self.device), \
                neg_rel.cuda(self.device)

            question = ' '.join(tokens)
            data = QAData([question, '', pos_index, '', len(tokens)])
            data.cand_rel = cand_rel
            data.cand_sub = ['']
            data.sub_rels = []
            yield Variable(seqs), seq_len, Variable(pos_rel1), Variable(pos_rel2), Variable(neg_rel1), Variable(neg_rel2), Variable(pos_rel), pos_len, Variable(neg_rel), neg_len, data


rel_sep_vocab = torch.load('../vocab/vocab.rel.sep.pt')
word_vocab = torch.load('../vocab/vocab.word&rel.pt')
file = '../others/valid.replace_ne.withpool'
loader = CandidateRankingLoader(file, word_vocab, rel_sep_vocab)
loader.next_question()
