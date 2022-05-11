# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 21:38:22 2022

@author: Dingyi
"""

import numpy as np
from torchsummary import summary
import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Positional Encoding from transformer
def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

# Positional Encoding + BiGRU
class ARPB(nn.Module):
    def __init__(self, d_pe, d_hidden):
        super(ARPB, self).__init__()
        self.d_pe = d_pe
        self.d_hidden = d_hidden

        self.pe_database = {}
        self.gru = nn.GRU(self.d_pe, d_hidden, 1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.d_hidden, 1, bias=False)

    def compute_pool_weights(self, lengths, features):
        max_len = int(lengths.max())
        pe_max_len = self.get_pe(max_len)
        pes = pe_max_len.unsqueeze(0).repeat(lengths.size(0), 1, 1).to(lengths.device)
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        pes = pes.masked_fill(mask == 0, 0)

        self.gru.flatten_parameters()
        packed = pack_padded_sequence(pes, lengths.cpu(), batch_first=True, enforce_sorted=True)
        out, _ = self.gru(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        out_emb, out_len = padded
        out_emb = (out_emb[:, :, :out_emb.size(2) // 2] + out_emb[:, :, out_emb.size(2) // 2:]) / 2
        scores = self.linear(out_emb)
        scores[torch.where(mask == 0)] = -10000

        weights = torch.softmax(scores / 0.1, 1)
        return weights, mask

    def forward(self, features, lengths):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        pool_weights, mask = self.compute_pool_weights(lengths, features)

        features = features[:, :int(lengths.max()), :]
        sorted_features = features.masked_fill(mask == 0, -10000)
        sorted_features = sorted_features.sort(dim=1, descending=True)[0]
        sorted_features = sorted_features.masked_fill(mask == 0, 0)

        pooled_features = (sorted_features * pool_weights).sum(1)
        return pooled_features, pool_weights

    def get_pe(self, length):
        """

        :param length: the length of the sequence
        :return: the positional encoding of the given length
        """
        length = int(length)
        if length in self.pe_database:
            return self.pe_database[length]
        else:
            pe = positional_encoding_1d(self.d_pe, length)
            self.pe_database[length] = pe
            return pe

def max_pooling(input):
    output = torch.max(input, axis=1)[0]
    return output

def mean_pooling(input):
    output = torch.mean(input, axis=1)
    return output

def LSE_pooling(input):
    output = torch.log(torch.mean(torch.exp(input), axis=1))
    return output

def choice_pooling(input, pooling_mode):
    if pooling_mode == 'max':
        return max_pooling(input).view(1,-1)
    elif pooling_mode == 'lse':
        return LSE_pooling(input).view(1,-1)
    elif pooling_mode == 'ave':
        return mean_pooling(input).view(1,-1)
    else:
        return input.view(1,-1)
    
class FeaturePooling(nn.Module):
    def __init__(self, pooling_mode='max'):
        super(FeaturePooling, self).__init__()
        self.pooling_mode = pooling_mode
        
    def forward(self, input):
        output = choice_pooling(input, self.pooling_mode)  
        return output

if __name__ == "__main__":
    gpool = GPO(32, 32)
    features = np.random.random((4, 3, 16))
    features = torch.tensor(features)
    # # print(features)
    image_lengths = [3, 3, 3, 3]
    image_lengths = torch.tensor(image_lengths)
    features, pool_weights = gpool(features, image_lengths)
    print(features)
    print(pool_weights)
    
    # # Find total parameters and trainable parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    
