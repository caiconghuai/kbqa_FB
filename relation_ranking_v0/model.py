#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2017-11-09

from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch
import sys
sys.path.append('../tools')
from embedding import Embeddings

class RelationRanking(nn.Module):

    def __init__(self, word_vocab, rel_vocab, config):
        super(RelationRanking, self).__init__()
        self.config = config
        self.word_embed = Embeddings(word_vec_size=config.d_word_embed, dicts=word_vocab)
        self.rel_embed = Embeddings(word_vec_size=config.d_rel_embed, dicts=rel_vocab)
#        print(self.rel_embed.word_lookup_table.weight.data)
        #rel_embed的初始化待改 rel_embed.lookup_table.weight.data.normal_(0, 0.1)

        if self.config.rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size=config.d_word_embed, hidden_size=config.d_hidden,
                              num_layers=config.n_layers, dropout=config.dropout_prob,
                              bidirectional=config.birnn)
        else:
            self.rnn = nn.LSTM(input_size=config.d_word_embed, hidden_size=config.d_hidden,
                               num_layers=config.n_layers, dropout=config.dropout_prob,
                               bidirectional=config.birnn)

        self.dropout = nn.Dropout(p=config.dropout_prob)
        seq_in_size = config.d_hidden
        if self.config.birnn:
            seq_in_size *= 2

        self.seq_out = nn.Sequential(
                        nn.Linear(seq_in_size, seq_in_size),
                        nn.BatchNorm1d(seq_in_size),
                        self.dropout,
                        nn.Linear(seq_in_size, config.d_rel_embed)
        )

    def match_score(self, seq_vec, x1, x2):
        """
        Score with Dot
        :param seq_vec: (batch, dim)
        :param x1: (batch, dim)
        :param x2: (neg_size, batch, dim)
        :return: (neg_size, batch)
        """
        pos_score = torch.sum(seq_vec * x1, 1)
        pos_score = pos_score.unsqueeze(0).expand(x2.size(0), x2.size(1))
        seq_vec = seq_vec.unsqueeze(0).expand_as(x2)
        neg_score = torch.sum(seq_vec * x2, 2)
        return pos_score, neg_score

    def forward(self, batch):
        # shape of batch (sequence length, batch size)
        seqs = batch[0]
        pos_rel = batch[1]
        neg_rel = batch[2]
        inputs = self.word_embed.forward(seqs) # shape (sequence length, batch_size, dimension of embedding)
        pos_rel_embed = self.rel_embed.word_lookup_table(pos_rel) # shape(batch_size, dimension of rel embedding)
#        pos_rel_embed = self.dropout(pos_rel_embed)
        neg_rel_embed = self.rel_embed.word_lookup_table(neg_rel) # shape(neg_size, batch_size, dimension of rel embedding)
#        neg_rel_embed = self.dropout(neg_rel_embed)

        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        if self.config.rnn_type.lower() == 'gru':
            h0 = autograd.Variable(inputs.data.new(*state_shape).zero_())
            outputs, ht = self.rnn(inputs, h0)
        else:
            h0 = c0 = autograd.Variable(inputs.data.new(*state_shape).zero_())
            outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        # shape of `outputs` - (sequence length, batch size, hidden size X num directions)
        # shape of `encoder` - (batch size, hidden size X num directions)
        # 感觉跟CFO的select层不太一样？取最后一层和第一层拼起来，而不是最后两层
        encoder = ht[-1] if not self.config.birnn else ht[-2:].transpose(0,1).contiguous().view(batch_size, -1)
        seq_encode = self.seq_out(encoder)
        # shape of `scores` - (neg_size, batch_size)
        scores = self.match_score(seq_encode, pos_rel_embed, neg_rel_embed)
        return scores

