#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'checkpoint management'

import os
import torch

name = 'model_parameters.pth.tar'

def save_checkpoint(state, address):
    folder = os.path.exists(address)
    if not folder:
        os.mkdir(address)
        print('--- create a new folder ---')
    fulladress = address + '\\' + name
    torch.save(state, fulladress)
    #print('model saved:', fulladress)

def load_checkpoint(address):
    fulladress = address + '\\' + name
    return torch.load(fulladress)