#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:44:24 2017

@author: reguig
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


class LogisticLoss(nn.Module):
    
    def __init__(self):
        super(LogisticLoss, self).__init__()
        
    def forward(self, output, target):
        loss = - target * output.sigmoid().log() - (1 - target) * (1 -output.sigmoid()).log()
        return loss.mean()
    
    
