#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' Linear regression module '

__author__ = 'Ma Cong'

import torch
import torch.nn.functional as F

class Net(torch.nn.Module):

    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.sigmoid(self.hidden(x))
        y = self.predict(x)
        return y