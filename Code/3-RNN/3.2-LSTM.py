# Author:StevenChaoo
# -*- coding:UTF-8 -*-

import torch
from torch import nn


class LSTM(nn.Module):
    '''
    Build LSTM model.

    Var:
        input_size:  The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers:  Number of recurrent layers
        batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
    '''

    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=50,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        print('lstm out size:')
        print(r_out.shape)
        out = self.out(r_out[:, -1, :])
        return out
