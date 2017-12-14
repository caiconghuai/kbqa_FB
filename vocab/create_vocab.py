#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2017-11-05

from dictionary import Dictionary
import torch
import pickle

def load_word_dictionary(filename, word_dict=None):
    if word_dict is None:
        word_dict = Dictionary()
        word_dict.add_unk_token()
        word_dict.add_pad_token()
    with open(filename) as f:
        for line in f:
            if not line:break
            line = line.strip()
            if not line:continue
            word_dict.add(line)
    return word_dict

word_vocab = load_word_dictionary('word.glove100k.txt')
torch.save(word_vocab, 'vocab.word.pt')

rel_vocab = load_word_dictionary('../freebase_data/FB5M.rel.txt')
torch.save(rel_vocab, 'vocab.rel.pt')

ent_vocab = load_word_dictionary('../freebase_data/FB5M.ent.txt')
torch.save(ent_vocab, 'vocab.ent.pt')

def load_type_dictionary(filename, word_dict=None):
    if word_dict is None:
        word_dict = Dictionary()
        word_dict.add_unk_token()
        word_dict.add_pad_token()
    data = pickle.load(open(filename, 'rb'))
    for ty in data:
        word_dict.add(ty)
    return word_dict

type_vocab = load_type_dictionary('../freebase_data/type.top-500.pkl')
torch.save(type_vocab, 'vocab.type.pt')

def load_rel_separated_dictionary(filename):
    rel1_dict = Dictionary()
    rel1_dict.add_unk_token()
    rel1_dict.add_pad_token()
    rel2_dict = Dictionary()
    rel2_dict.add_unk_token()
    rel2_dict.add_pad_token()
    with open(filename) as f:
        for line in f:
            if not line:break
            line = line.strip()
            if not line:continue
            line = line.split('.')
            rel1 = '.'.join(line[:-1])
            rel2 = line[-1]
            rel1_dict.add(rel1)
            rel2_dict.add(rel2)
    return rel1_dict, rel2_dict

rel1_vocab, rel2_vocab = load_rel_separated_dictionary('../freebase_data/FB5M.rel.txt')
torch.save((rel1_vocab, rel2_vocab), 'vocab.rel.sep.pt')
