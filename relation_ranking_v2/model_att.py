#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2017-12-13

from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch
import sys
sys.path.append('../tools')
from embedding import Embeddings
from attention import MLPWordSeqAttention

class RelationRanking(nn.Module):

    def __init__(self, word_vocab, rel_vocab, config):
        super(RelationRanking, self).__init__()
        self.config = config
        rel1_vocab, rel2_vocab = rel_vocab
        self.word_embed = Embeddings(word_vec_size=config.d_word_embed, dicts=word_vocab)
#        self.word_embed2 = Embeddings(word_vec_size=config.d_word_embed, dicts=word_vocab)
        self.rel1_embed = Embeddings(word_vec_size=config.d_rel_embed, dicts=rel1_vocab)
        self.rel2_embed = Embeddings(word_vec_size=config.d_rel_embed, dicts=rel2_vocab)
#        print(self.rel_embed.word_lookup_table.weight.data)
        #rel_embed的初始化待改 rel_embed.lookup_table.weight.data.normal_(0, 0.1)

        if self.config.rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size=config.d_word_embed, hidden_size=config.d_hidden,
                              num_layers=config.n_layers, dropout=config.dropout_prob,
                              bidirectional=config.birnn,
                              batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=config.d_word_embed, hidden_size=config.d_hidden,
                               num_layers=config.n_layers, dropout=config.dropout_prob,
                               bidirectional=config.birnn,
                               batch_first=True)

        self.dropout = nn.Dropout(p=config.dropout_prob)
        seq_in_size = config.d_hidden
        if self.config.birnn:
            seq_in_size *= 2

        self.question_attention = MLPWordSeqAttention(input_size=config.d_rel_embed, seq_size=seq_in_size)

        self.bilinear = nn.Bilinear(seq_in_size, config.d_rel_embed, 1, bias=False)

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

    def question_encoder(self, inputs, rel_embed, pos=None):
        '''
        :param inputs: (batch, dim1)
        :param rel_embed: (batch, dim2) or (neg_size, batch, dim2)
        return: (batch, 1)
        '''
        batch_size = inputs.size(0)
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        if self.config.rnn_type.lower() == 'gru':
            h0 = autograd.Variable(inputs.data.new(*state_shape).zero_())
            outputs, ht = self.rnn(inputs, h0)
        else:
            h0 = c0 = autograd.Variable(inputs.data.new(*state_shape).zero_())
            outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
            outputs.contiguous()
        # shape of `outputs` - (batch size, sequence length, hidden size X num directions)
        # shape of `encoder` - (batch size, hidden size X num directions)
#        encoder = ht[-1] if not self.config.birnn else ht[-2:].transpose(0,1).contiguous().view(batch_size, -1)
#        seq_encode = self.seq_out(encoder)
        if pos:
            neg_size = pos
        else: # neg的要扩展
            neg_size, batch_size, embed_size = rel_embed.size()
            seq_len, seq_emb_size = outputs.size()[1:]
            outputs = outputs.unsqueeze(0).expand(neg_size, batch_size, seq_len,
                            seq_emb_size).contiguous().view(neg_size*batch_size, seq_len, -1)
            rel_embed = rel_embed.view(neg_size * batch_size, -1)
        # `seq_encode` - (batch, hidden size X num directions)
        # `weight` - (batch, length)
        seq_encode, weight = self.question_attention.forward(rel_embed, outputs)
        # `score` - (batch, 1)
        score = self.bilinear(seq_encode, rel_embed)

        if pos:  # pos要把结果扩展
            score = score.squeeze(1).unsqueeze(0).expand(neg_size, batch_size)
        else:
            score = score.view(neg_size, batch_size)
        return score

    def forward(self, batch):
        seqs, pos_rel1, pos_rel2, neg_rel1, neg_rel2 = batch
        # shape of seqs (batch size, sequence length)
        seqs = torch.transpose(seqs, 0, 1)
        # shape (batch_size, sequence length, dimension of embedding)
        inputs = self.word_embed.forward(seqs)
        # shape (batch_size, dimension of rel embedding)
        pos_rel1_embed = self.rel1_embed.word_lookup_table(pos_rel1)
        pos_rel2_embed = self.rel2_embed.word_lookup_table(pos_rel2)
#        pos_rel_embed = self.dropout(pos_rel_embed)
        # shape (neg_size, batch_size, dimension of rel embedding)
        neg_rel1_embed = self.rel1_embed.word_lookup_table(neg_rel1)
        neg_rel2_embed = self.rel2_embed.word_lookup_table(neg_rel2)
#        neg_rel_embed = self.dropout(neg_rel_embed)

        neg_size = neg_rel1_embed.size(0)
        # shape of `score` - (neg_size, batch_size)
        pos_score1 = self.question_encoder(inputs, pos_rel1_embed, neg_size)
        pos_score2 = self.question_encoder(inputs, pos_rel2_embed, neg_size)
        neg_score1 = self.question_encoder(inputs, neg_rel1_embed)
        neg_score2 = self.question_encoder(inputs, neg_rel2_embed)
        return pos_score1, pos_score2, neg_score1, neg_score2