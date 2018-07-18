#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' test module '

__author__ = 'Ma Cong'

import torch
import numpy as np
from torch.autograd import Variable

from net import Net
import checkpoint as cp

class Test(object):
    def __init__(self, n_in, n_hidden, n_out, checkpoint):
        self.net = Net(n_in, n_hidden, n_out)
        self.checkpoint_path = checkpoint

    def test(self, x):
        checkpoint = cp.load_checkpoint(address=self.checkpoint_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        x_min, x_max, y_min, y_max = checkpoint['norm_dict']
        x = (x - x_min) / (x_max - x_min)
        with torch.no_grad():
            x = torch.from_numpy(x)
            y_t = self.net(x)
            y_t = y_t.numpy() * (y_max - y_min) + y_min
            print(y_t)