#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'create models'

__author__ = 'Ma Cong'

import torch
import torch.nn as nn
import torch.nn.functional as F

lay = [4, 200, 200, 3]

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(lay[0], lay[1]),
            nn.BatchNorm1d(lay[1]),
            torch.nn.ReLU(),
            #torch.nn.Dropout(p=0.5),
            torch.nn.Linear(lay[1], lay[2]),
            nn.BatchNorm1d(lay[2]),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(lay[2], lay[3]),
        )

    def forward(self, input):
        x = self.Classes(input)
        return x