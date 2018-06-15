#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'test'

__author__ = 'Ma Cong'

import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np

import checkpoint as cp
import model as mdl

net = mdl.net()
df = pd.read_csv('new_wc2018.csv', encoding='gbk')
df.drop(index=0, axis=0)
x = np.array(df.ix[:, 3:7], dtype=np.float32).tolist()
x = torch.FloatTensor(x)
x = Variable(x)

def main():
    checkpoint = cp.load_checkpoint(address='parameters.pth')
    net.load_state_dict(checkpoint['state_dict'])
    outputs = net(x)
    outputs = outputs.cpu()
    outputs = outputs.data.numpy()

    pred_choice = []
    for out in outputs:
        K = 1
        index = np.argpartition(out, -K)[-K:]
        pred_choice.append(index)
    pre = np.array(pred_choice)

    df['score'] = pre
    df.to_csv('predict.csv', encoding='gbk')

if __name__ == '__main__':
    main()