#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2018-01-12

from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import sys
import numpy as np
sys.path.append('../tools')
sys.path.append('../vocab')
from embedding import Embeddings

class RelationRanking(nn.Module):
    def __init__(self, word_vocab, config):
        super(RelationRanking, self).__init__()
        self.config = config
        self.word_embed = Embeddings(word_vec_size=config.d_word_embed, dicts=word_vocab)

        self.conv = nn.Sequential(
            nn.Dropout(p=config.dropout_prob),
            nn.Conv2d(1, config.channel_size, (config.conv_kernel_1, config.conv_kernel_2), stride=1,
                      padding=(config.conv_kernel_1//2, config.conv_kernel_2//2)), #channel_in=1, channel_out=8, kernel_size=3*3
            nn.ReLU(True))

        self.seq_maxlen = config.seq_maxlen + (config.conv_kernel_1 + 1) % 2
        self.rel_maxlen = config.rel_maxlen + (config.conv_kernel_2 + 1) % 2
        p_size1 = self.seq_maxlen // config.pool_kernel_1
        p_size2 = self.rel_maxlen // config.pool_kernel_2

        self.pooling = nn.MaxPool2d((config.pool_kernel_1, config.pool_kernel_2),
                                    stride=(config.pool_kernel_1, config.pool_kernel_2), padding=0)

        self.fc = nn.Sequential(
            nn.Linear(p_size1*p_size2*config.channel_size, 20),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_prob),
            nn.Linear(20, 1),
            nn.Sigmoid())

    def matchPyramid(self, seq, rel, seq_len, rel_len):
        '''
        param:
            seq: (batch, _seq_len, embed_size)
            rel: (batch, _rel_len, embed_size)
            seq_len: (batch,)
            rel_len: (batch,)
        return:
            score: (batch, 1)
        '''
        batch_size = seq.size(0)

        rel_trans = torch.transpose(rel, 1, 2)
        # (batch, 1, seq_len, rel_len)
 #       cross = torch.bmm(seq, rel_trans).unsqueeze(1)
        # 将内积改为cos 相似度
        seq_norm = torch.sqrt(torch.sum(seq*seq, dim=2, keepdim=True))
        rel_norm = torch.sqrt(torch.sum(rel_trans*rel_trans, dim=1, keepdim=True))
        cross = torch.bmm(seq/seq_norm, rel_trans/rel_norm).unsqueeze(1)

#        print('cross: ', cross.size())
#        print(cross.squeeze(1).squeeze(0))

        # (batch, channel_size, seq_len, rel_len)
        conv1 = self.conv(cross)
        channel_size = conv1.size(1)
#        print('conv: ', conv1.size())

        # (batch, seq_maxlen)
        # (batch, rel_maxlen)
        dpool_index1, dpool_index2 = self.dynamic_pooling_index(seq_len, rel_len, self.seq_maxlen,
                                                                self.rel_maxlen)
        dpool_index1 = dpool_index1.unsqueeze(1).unsqueeze(-1).expand(batch_size, channel_size,
                                                                self.seq_maxlen, self.rel_maxlen)
        dpool_index2 = dpool_index2.unsqueeze(1).unsqueeze(2).expand_as(dpool_index1)
#        print('d1: ', dpool_index1.size())
#        print('d2: ', dpool_index2.size())
        conv1_expand = torch.gather(conv1, 2, dpool_index1)
        conv1_expand = torch.gather(conv1_expand, 3, dpool_index2)
#        print(conv1_expand.size())

        # (batch, channel_size, p_size1, p_size2)
        pool1 = self.pooling(conv1_expand).view(batch_size, -1)
#        print('pool: ', pool1.size())

        # (batch, 1)
        out = self.fc(pool1)
        return out

    def dynamic_pooling_index(self, len1, len2, max_len1, max_len2):
        def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):
            stride1 = 1.0 * max_len1 / len1_one
            stride2 = 1.0 * max_len2 / len2_one
            idx1_one = [int(i/stride1) for i in range(max_len1)]
            idx2_one = [int(i/stride2) for i in range(max_len2)]
#            mesh1, mesh2 = np.meshgrid(idx1_one, idx2_one)
#            index_one = np.transpose(np.stack([np.ones(mesh1.shape) * batch_idx, mesh1, mesh2]), (2,1,0))
            return idx1_one, idx2_one
        batch_size = len(len1)
        index1, index2 = [], []
        for i in range(batch_size):
            idx1_one, idx2_one = dpool_index_(i, len1[i], len2[i], max_len1, max_len2)
            index1.append(idx1_one)
            index2.append(idx2_one)
#        print(index1)
#        print(index2)
        index1 = torch.LongTensor(index1)
        index2 = torch.LongTensor(index2)
        if self.config.cuda:
            index1 = index1.cuda()
            index2 = index2.cuda()
        return Variable(index1), Variable(index2)

    def forward(self, batch):
        seqs, seq_len, pos_rel1, pos_rel2, neg_rel1, neg_rel2, pos_rel, pos_rel_len, neg_rel, neg_rel_len = batch
        neg_size, batch, neg_len = neg_rel.size()

        # (batch, len, emb_size)
        seqs_embed = self.word_embed.forward(seqs)

        # (batch, len, emb_size)
        pos_embed = self.word_embed.forward(pos_rel)
        # (batch, 1)
        pos_score = self.matchPyramid(seqs_embed, pos_embed, seq_len, pos_rel_len)
        # (neg_size, batch)
        pos_score = pos_score.squeeze(-1).unsqueeze(0).expand(neg_size, batch)

        # (neg_size*batch, len, emb_size)
        neg_embed = self.word_embed.forward(neg_rel.view(-1, neg_len))
        seqs_embed = seqs_embed.unsqueeze(0).expand(neg_size, batch, seqs_embed.size(1),
                    seqs_embed.size(2)).contiguous().view(-1, seqs_embed.size(1), seqs_embed.size(2))
        # (neg_size*batch,)
        neg_rel_len = neg_rel_len.view(-1)
        seq_len = seq_len.unsqueeze(0).expand(neg_size, batch).contiguous().view(-1)
        # (neg_size*batch, 1)
        neg_score = self.matchPyramid(seqs_embed, neg_embed, seq_len, neg_rel_len)
        # (neg_size, batch)
        neg_score = neg_score.squeeze(-1).view(neg_size, batch)

        return pos_score, neg_score

if __name__ == '__main__':

    x1_len = torch.Tensor([5,6,7,4])
    x2_len = torch.Tensor([6,4,5,8])
    from args import get_args
    config = get_args()
    word_vocab = torch.load(config.vocab_file)
    RelationRanking(word_vocab, config).dynamic_pooling_index(x1_len, x2_len, 8, 8)

