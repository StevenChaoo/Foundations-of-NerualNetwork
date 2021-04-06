# Author:StevenChaoo
# -*- coding:UTF-8 -*-

import torch
from torch import nn


class GRU(nn.Module):
    '''
    Build GRU model.

    Var:
        input_size:  The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
    '''

    def __init__(self, output_num):
        super(GRU, self).__init__()
        self.hidden_size = 64
        self.gru = nn.GRU(
            input_size=50,
            hidden_size=64,
            batch_first=True
        )
        self.out = nn.Linear(64, output_num)
        self.hidden = None

    def forward(self, x):
        x, self.hidden = self.gru(x)
        x = self.out(x)
        return x, self.hidden
