#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 10:01:08 2017

@author: reguig
"""

from cnn import DeepCNNAdaptiveStride
from wrapper import SkLearnTorch
from dataFunctions import getData
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

import pandas as pd

import sys 
import csv
import numpy as np

if sys.argv[1]:
    nameDataset = sys.argv[1]
    nameDataset = nameDataset.replace('/home/reguig/datasets/UCR/', '')
    print("Dataset : %s"%(nameDataset))
    trainx, trainy, testx, testy = getData(nameDataset)
    size = trainx.size(1)
    negativeClass = -1 in (trainy.data.numpy())
    if negativeClass : 
        trainy = trainy + 1
        testy = testy +  1
    
    kernel_sizes = [[int(0.05 * size), int(0.05*size)], [int(0.1 * size), int(0.05 * size)],
                     [int(0.2 * size), int(0.05 * size)]]
    
    nbConvLayers = [2]
    
    strides = [[1, 1], 
               [1, max(int(0.01 * size), 1)], 
               [max(int(0.01 * size),1), 1], 
               [max(int(0.01 * size),1), max(int(0.01 * size),1)], 
               [1, max(int(0.02 * size), 1)], 
               [max(int(0.02 * size), 1), 1], 
               [max(int(0.02 * size), 1), max(int(0.02 * size), 1)]]
    
    paddings = [[0,0]]
    
    k = [int(0.05 * size),  int(0.15 * size), int(0.2 * size)]
    
    weightDecays = [ 0.001, 0.01, 0.1]
    
    parameters = {"k":k, "kernel_sizes" : kernel_sizes, "strides" : strides,
                  'weight_decay' : weightDecays, 'nbConvLayers' : nbConvLayers}
    fit_params = {"epochs" : 5000, 'lr' : 0.01, 'lr_decay' : .0001}
    
    model = SkLearnTorch(cls = DeepCNNAdaptiveStride)
    skf = StratifiedKFold(n_splits = 3)
    splits = list(skf.split(trainx.data.numpy(), trainy.data.numpy()))
    
    clf = GridSearchCV(model, parameters, n_jobs =  1, cv = splits, fit_params = fit_params)
    
    clf.fit(trainx.data.numpy(),trainy.data.numpy())
    results = pd.DataFrame.from_dict(clf.cv_results_)
    results.to_html("/home/reguig/ResultatsUCR/ResultatsHTML/DeepCNN/%s.html"%(nameDataset))
    
    tmp = results[['params','mean_train_score','std_train_score', 'mean_test_score', 'std_test_score']].sort_values('mean_test_score', ascending = False)
    writer = pd.ExcelWriter('/home/reguig/ResultatsUCR/ResultatsExcel/DeepCNN/%s.xlsx'%(nameDataset))
    #writer = pd.ExcelWriter('Beef.xlsx')
    #tmp.to_excel(writer,'%s'%(nameDataset))
    
    tmp.to_excel(writer,'%s'%(nameDataset))
    writer.save()
    testScore = clf.best_estimator_.score(testx.data.numpy(), testy.data.numpy())
    trainScore = clf.best_estimator_.score(trainx.data.numpy(), trainy.data.numpy())
    f = open("/home/reguig/ResultatsUCR/ResultatsTest/DeepCNN/scoresTests.csv", "a")
    c = csv.writer(f)
    c.writerow([nameDataset, round(testScore, 3), round(trainScore, 3), clf.best_params_])
    print("Score en train : %f \nScore en test : %f"%(trainScore, testScore))