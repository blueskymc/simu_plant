#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' data module '

__author__ = 'Ma Cong'

import numpy as np
import pandas as pd

def get_samples(path, num_in=3, num_out=1):
    df_train = pd.read_csv(path, encoding='gbk')
    samples = df_train.values
    assert samples.shape[1] == (num_in + num_out), '输入层和输出层之和不等于csv文件的列数'
    x = samples[:, :num_in]
    y = samples[:, num_in:]
    # print(x.shape)
    # print(y.shape)
    return x.astype(np.float32), y.astype(np.float32)
