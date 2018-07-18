#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' main module '

__author__ = 'Ma Cong'

import numpy as np
import matplotlib.pyplot as plt
import argparse

from data_utils import get_samples
import train
from test import Test

def main():
    '''main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', default='train', help='train or test')
    parser.add_argument('--n_in', default=3, type=int, help='输入层大小')
    parser.add_argument('--n_hide', default=5, type=int, help='隐藏层大小')
    parser.add_argument('--n_out', default=1, type=int, help='输出层大小')
    parser.add_argument('--epoch', default=100000, type=int, help='训练次数')
    parser.add_argument('--lr', default=0.001, help='学习速率')
    parser.add_argument('--data', default='train.csv', help='训练数据集')
    parser.add_argument('--checkpoint', default='checkpoints', help='参数保存位置')
    opt = parser.parse_args()
    print(opt)


    if opt.state == 'train':
        model = train.train_net(opt.n_in, opt.n_hide, opt.n_out, opt.checkpoint,
                                opt.epoch, opt.lr)
        x, y = get_samples(opt.data, opt.n_in, opt.n_out)
        model.train(x, y)

    elif opt.state == 'test':
        test = Test(opt.n_in, opt.n_hide, opt.n_out, opt.checkpoint)
        # 14.61,13.49,22.67,17.81
        x = np.array([[14.61, 13.49, 22.67]], dtype=np.float32)
        test.test(x)

    else:
        print('Error state, must choose from train and eval!')

if __name__ == '__main__':
    main()
