#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 15:20:11 2017

@author: reguig
"""
"""
import torch
import torch.nn as nn 
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
"""
from torch.autograd import Function

"""
Fonction permettant de calculer la distance entre deux series de meme taille
"""

class DistanceShapelet(Function):
    
    def forward(self, shapelet, serie):
        length = shapelet.size()[1]
        diff = serie - shapelet
        result = diff ** 2
        result = result.mean()        
        self.length = length
        self.diff = diff.mean()
        return result
    
    def backward(self, grad_output):
        diff = self.diff
        length = self.length
        return grad_output * 2.0/length * diff
        
