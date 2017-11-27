#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2017-11-18
import sys, os, io
import pickle
import numpy as np
import torch
from torch.autograd import Variable
sys.path.append('../vocab')
sys.path.append('../tools')
import virtuoso

def create_seq_multi_label_data(batch_size, qa_data, word_vocab, type_vocab):
    file_type = qa_data.split('.')[-2]
    log_file = open('data/%s.subject_type.txt' %file_type, 'w')
    seqs = []
    seq_labels = []
    batch_index = -1     # the index of sequence batches
    seq_index = 0       # sequence index within each batch
    pad_index = word_vocab.lookup(word_vocab.pad_token)

    data_list = pickle.load(open(qa_data, 'rb'))
    for data in data_list:
        if not data.text_attention_indices:
            continue

        tokens = data.question.split()
        types = virtuoso.id_query_type(data.subject)
        log_file.write('%s\t%s\n' %(data.question, types))
        types = type_vocab.convert_to_index(types)

        if seq_index % batch_size == 0:
            seq_index = 0
            batch_index += 1
            seqs.append(torch.LongTensor(len(tokens), batch_size).fill_(pad_index))
            seq_labels.append(torch.FloatTensor(batch_size, len(type_vocab)).fill_(0))
            print('batch: %d' %batch_index)

        seqs[batch_index][0:len(tokens),seq_index] = torch.LongTensor(word_vocab.convert_to_index(tokens))
        for t in types:
            seq_labels[batch_index][seq_index, t] = 1
        seq_index += 1

    torch.save((seqs, seq_labels), 'data/%s.subject_type.pt' %file_type)


class SeqMultiLabelLoader():

    def __init__(self, infile, device=-1):
        self.seqs, self.seq_labels = torch.load(infile)
        self.batch_size = self.seqs[0].size(1)
        self.batch_num = len(self.seqs)

        if device >= 0:
            for i in range(self.batch_num):
                self.seqs[i] = Variable(self.seqs[i].cuda(device))
                self.seq_labels[i] = Variable(self.seq_labels[i].cuda(device))

    def next_batch(self, shuffle = True):
        if shuffle:
            indices = torch.randperm(self.batch_num)
        else:
            indices = range(self.batch_num)
        for i in indices:
            yield self.seqs[i], self.seq_labels[i]

if __name__ == '__main__':
    word_vocab = torch.load('../vocab/vocab.word.pt')
    type_vocab = torch.load('../vocab/vocab.type.pt')
    create_seq_multi_label_data(128, '../data/QAData.valid.pkl', word_vocab, type_vocab)
    create_seq_multi_label_data(128, '../data/QAData.test.pkl', word_vocab, type_vocab)
    create_seq_multi_label_data(128, '../data/QAData.train.pkl', word_vocab, type_vocab)
