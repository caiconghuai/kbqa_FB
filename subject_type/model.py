from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch
import sys
sys.path.append('../tools')
from embedding import Embeddings

class EntityType(nn.Module):

    def __init__(self, dicts, config, n_type):
        super(EntityType, self).__init__()
        self.config = config
        self.embed = Embeddings(word_vec_size=config.d_embed, dicts=dicts)
        if self.config.rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size=config.d_embed, hidden_size=config.d_hidden,
                              num_layers=config.n_layers, dropout=config.dropout_prob,
                              bidirectional=config.birnn)
        else:
            self.rnn = nn.LSTM(input_size=config.d_embed, hidden_size=config.d_hidden,
                               num_layers=config.n_layers, dropout=config.dropout_prob,
                               bidirectional=config.birnn)

        self.dropout = nn.Dropout(p=config.dropout_prob)
        seq_in_size = config.d_hidden
        if self.config.birnn:
            seq_in_size *= 2

        self.hidden2tag = nn.Sequential(
                        nn.Linear(seq_in_size, seq_in_size),
                        nn.BatchNorm1d(seq_in_size),
                        self.dropout,
                        nn.Linear(seq_in_size, n_type),
                        nn.Sigmoid()
        )

    def forward(self, batch):
        # shape of batch (sequence length, batch size)
        inputs = self.embed.forward(batch[0]) # shape (sequence length, batch_size, dimension of embedding)
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
        encoder = ht[-1] if not self.config.birnn else ht[-2:].transpose(0,1).contiguous().view(batch_size, -1)
        tags = self.hidden2tag(encoder)
        # print(tags)
        return tags

