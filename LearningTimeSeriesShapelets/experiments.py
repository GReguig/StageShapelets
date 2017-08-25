#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:11:12 2017

@author: reguig
"""

from cnn import AdaptiveStrideCNN
from ShapeletsOps import LogisticShapeletsLearner
from wrapper import SkLearnTorch
from dataFunctions import getData, ToOneHot
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import torch
import pandas as pd

import sys 
import csv
import numpy as np
import matplotlib.pyplot as plt
import math

"""
class LogisticShapeletsLearner(nn.Module):

    def __init__(self, nb_shapelets = 15, shapelets_size = 10, scale = 3 , alpha = -30, weight_decay = 0.0):
        
"""
if sys.argv[1]:
    nameDataset = sys.argv[1]
    nameDataset = nameDataset.replace('/home/reguig/datasets/UCR/', '')
    print("Dataset : %s"%(nameDataset))
    
    trainx, trainy, testx, testy = getData(nameDataset, oneHot = False)
    size = trainx.shape[1]
    nbClasses = len(np.unique(trainy))
    
    ######################################## PARAMS CNN #########################################    
    
    kernel_size = [10, 20, 30]
    stride = [1, 15, 20]
    k = [ 5, 7, 10]
    nb_filters = [5, 10, 15, 20]
    weightDecays = [ 0.0, 0.001, 0.0001, 0.00001]
    
    
    parameters = {"k":k, "kernel_size" : kernel_size, "stride" : stride,'weight_decay' : weightDecays, 
                  'nb_filters' : nb_filters}
    
    ##################################################### LTS ###################
    """
    nb_shapelets = [int (np.log(sum([size - i + 1 for i in range(1, size)])) * (nbClasses - 1))]
    shapelets_size = [int(0.1 * size), int(0.2 * size)]
    scale = [2, 3]
    weight_decay = [0.0, 0.1, 0.01]
  
    
    parameters = {'nb_shapelets' : nb_shapelets, 'shapelets_size' : shapelets_size, 
                  'scale' : scale, 'weight_decay' :weight_decay}
    """
    
    fit_params = {"epochs" : 10000, 'lr' : 0.1}
    
    model = SkLearnTorch()
    skf = StratifiedKFold(n_splits = 3)
    splits = list(skf.split(trainx, trainy ))
    
    ################################################# G R I D  S E A R C H ##################################
    
    clf = GridSearchCV(model, parameters, n_jobs = -1, cv = splits, fit_params = fit_params, refit=True)
    clf.fit(trainx,trainy)
    
    ############################################# R E S U L T A T S ####################################
    
    results = pd.DataFrame.from_dict(clf.cv_results_)
    # Meilleurs parametres
    best_params = list(results[results['rank_test_score']==1]['params'])
    print("%d configuration(s) trouvee(s) pour les meilleurs parametres"%(len(best_params)))
    results.to_html("/home/reguig/ResultatsUCR/ResultatsHTML/CNN/%s.html"%(nameDataset))
    
    model = SkLearnTorch()
    best_performances = []
    best_performancesTest = []
    print("Tests des meilleures configurations trouvees\n")
    for param in best_params : 
        print("Test de la configuration : ")
        print(param)
        model.set_params(**param)
        tmpPerfs = []
        tmpPerfsTest = []
        for j in range(10) : 
            print("Modele %d / 10 : \n"%(j))
            model.fit(trainx, trainy, epochs = 10000, lr=0.001) 
            tmpPerfs.append(round(model.score(trainx,trainy), 3))
            tmpPerfsTest.append(round(model.score(testx, testy), 3))
        best_performances.append(tmpPerfs)
        best_performancesTest.append(tmpPerfsTest)
        
    ######### PERFORMANCES EN TEST    
    best_performancesTest = np.asarray(best_performancesTest)
    meanTest = best_performancesTest.mean(axis=1)
    stdevTest = best_performancesTest.std(axis = 1)
    
    ######## PERFORMANCES EN TRAIN
    best_performances = np.asarray(best_performances)    
    mean = best_performances.mean(axis = 1)
    stdev = best_performances.std(axis = 1)
    
    df = pd.DataFrame({'params': best_params , 'mean' : mean, 'stdev' : stdev,
                       'meanTest' : meanTest, 'stdevTest' : stdevTest}).sort_values('mean', ascending = False)
    w2 = pd.ExcelWriter('/home/reguig/ResultatsUCR/ResultatsExcelBestParams/CNN/%s_Best.xlsx'%(nameDataset))
    df.to_excel(w2, '%s_Best'%(nameDataset))
    w2.save()
    #Selection
    
    tmp = results[['params','mean_train_score','std_train_score', 'mean_test_score', 'std_test_score']].sort_values('mean_test_score', ascending = False)
    writer = pd.ExcelWriter('/home/reguig/ResultatsUCR/ResultatsExcel/CNN/%s.xlsx'%(nameDataset))
    
    
    tmp.to_excel(writer,'%s'%(nameDataset))
    
    writer.save()
    
    
"""
## SCORE ##
plt.figure()
plt.errorbar(results['param_k'], results['mean_test_score'], yerr=results['std_test_score'], label='test')
plt.errorbar(results['param_k'], results['mean_train_score'],yerr=results['std_train_score'], label = 'train')
plt.legend()
plt.title("Accuracy en fonction du kmax")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.savefig("AccuracyKmax.png")
#Losss

plt.figure()
plt.errorbar(results['param_kernel_size'], results['mean_test_score'], yerr=results['std_test_score'], label='test')
plt.errorbar(results['param_kernel_size'], results['mean_train_score'],yerr=results['std_train_score'], label = 'train')
plt.legend()
plt.title("Accuracy en fonction de la taille du noyau de convolution")
plt.xlabel("Taille du kernel")
plt.ylabel("Accuracy")
plt.savefig("AccuracyKernel.png")
"""
"""
trainx, trainy, testx, testy = getData("Beef")    
#trainy, testy = trainy + 1, testy + 1
#trainy, testy = oneHot(trainy.data.numpy()), oneHot(testy.data.numpy())
#trainy, testy = trainy.long(), testy.long()
#testy[testy == 0] = -1
#net = LogisticRegShapelets(nb_shapelets = 2, tailleShapelets = 5, series = trainx, nb_classes = 2, alpha = -30)
#net = CNNShapelets(nb_shapelets = nbShapelets, tailleShapelets = taille, scale=3, alpha=-30)    
 #net = SimpleCNN()
sklearn = SkLearnTorch(SimpleCNN)

########################### G R I D   S E A R C H ##############################

skf = StratifiedKFold(n_splits = 3)
gen = skf.split(trainx.data.numpy(), trainy.data.numpy())
splits = list(gen)
parameters = {"k" : [2, 3]}
fit_params = {"epochs" : 500, 'lr':0.001}
clf = GridSearchCV(sklearn, parameters, n_jobs = -1, cv = splits, fit_params = fit_params)
res = clf.fit(trainx.data.numpy(),trainy.data.numpy())
"""
