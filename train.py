#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' train module '

__author__ = 'Ma Cong'

import torch
import torch.nn as nn
from torch.autograd import Variable

from net import Net
import checkpoint as cp

class train_net(object):
    def __init__(self, n_in, n_hidden, n_out, checkpoint, epoch=1000, lr=0.01):
        self.net = Net(n_in, n_hidden, n_out)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.n_epoch = epoch
        self.checkpoint_path = checkpoint

    def train(self, x, y):
        x_min, x_max = x.min(axis=0), x.max(axis=0)
        x = (x - x_min) / (x_max - x_min)
        y_min, y_max = y.min(axis=0), y.max(axis=0)
        y = (y - y_min) / (y_max - y_min)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        x, y = Variable(x), Variable(y)

        last_loss = 1
        for i in range(self.n_epoch):
            prediction = self.net(x)
            loss = self.criterion(prediction, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i + 1) % 1000 == 0:
                print('loss: %.8f' % loss)
                self.save_checkpoints((x_min, x_max, y_min, y_max))
                if loss > last_loss:
                    self.descent_lr()

            last_loss = loss

        # 训练完毕保存参数
        self.save_checkpoints((x_min, x_max, y_min, y_max))


    def save_checkpoints(self, norm_dict):
        checkpoint = {'state_dict': self.net.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'norm_dict': norm_dict}
        cp.save_checkpoint(checkpoint, address=self.checkpoint_path)

    def descent_lr(self, decay_rate=0.9):
        for param_group in self.optimizer.param_groups:
            if param_group['lr'] < 1e-6:
                print('learning rate is lower than 1e-6, no more descent!')
                return 0
            param_group['lr'] = param_group['lr'] * decay_rate
            print('descent learning rate to %.5f' % param_group['lr'])