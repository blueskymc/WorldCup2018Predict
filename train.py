#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'base class for train'

__author__ = 'Ma Cong'

import torch
from torch.autograd import Variable
from datetime import datetime

import checkpoint as cp
import match_data
import model as mdl

n_epoch = 1000

x, y, xt, yt = match_data.get_train_data()
x = torch.FloatTensor(x)
y = torch.LongTensor(y)
xt = torch.FloatTensor(xt)
yt = torch.LongTensor(yt)
x, y = Variable(x), Variable(y)
xt, yt = Variable(xt), Variable(yt)

net = mdl.net()
#optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=0.002)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)
loss_func = torch.nn.CrossEntropyLoss()

def now():
    return datetime.now().strftime('%c')

def train_one_epoch():
    net.train()

    print(now())
    print('Begin training...')

    prediction = net(x)
    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train(checkpoint_path):
    # 是否装载模型参数
    load = False

    if load:
        checkpoint = cp.load_checkpoint(address=checkpoint_path)
        net.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, n_epoch):
        train_one_epoch()

        # 保存参数
        checkpoint = {'epoch': epoch, 'state_dict': net.state_dict(),
                      'optimizer': optimizer.state_dict()}
        cp.save_checkpoint(checkpoint, address=checkpoint_path)

        eval()

def eval():
    correct = 0
    sum = 0
    outputs = net(xt)

    pred_choice = outputs.data.max(1)[1]
    correct += pred_choice.eq(yt.data).cpu().sum()
    sum += len(yt)
    print('correct/sum:%d/%d, %.4f' % (correct, sum, correct / sum))

def main():
    train(checkpoint_path='parameters.pth')

if __name__ == '__main__':
    main()