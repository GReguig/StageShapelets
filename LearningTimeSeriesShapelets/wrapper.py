#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 09:50:32 2017

@author: reguig
"""

from sklearn.base import BaseEstimator
#from Modules import LogisticRegShapelets, CNNShapelets
from dataFunctions import getData, ToOneHot
from cnn import AdaptiveStrideCNN, DeepCNNAdaptiveStride
from ShapeletsOps import LogisticShapeletsLearner
from torch.autograd import Variable

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

import torch
import numpy as np
import copy

class SkLearnTorch(BaseEstimator):
    
    def __init__(self, cls=AdaptiveStrideCNN, **kwargs):
        super(SkLearnTorch, self).__init__()
        self.cls = cls(**kwargs)
        
    def fit(self, x , y, **kwargs):
        acceptedTypes = [np.ndarray, list, np.core.memmap]
        
        if (type(x) not in acceptedTypes) or (type(y) not in acceptedTypes) : 
            print("Type non accepte")
            raise TypeError("X and Y must be arrays or lists found %s and %s"%(type(x), type(y)))
        self.binary = len(np.unique(y)) == 2
            
        if self.binary :
            self.cls.fit(Variable(torch.Tensor(x)), Variable(torch.Tensor(y)),**kwargs)
        else : 
            self.cls.fit(Variable(torch.Tensor(x)), Variable(torch.Tensor(ToOneHot(y))),**kwargs)
        return self
    
    def score(self, x, y):
        data = Variable(torch.Tensor(x))
        if self.binary : 
            targets = Variable(torch.Tensor(y))
        else :     
            targets = Variable(torch.Tensor(ToOneHot(y)))
        return self.cls.score(data, targets)
    
    def clone(self):
        return copy.deepcopy(self)
    
    def predict(self, x):
        return self.cls.predict(x)
    
    def get_params( self, deep = True):
        return self.cls.get_params(deep)
    
    def set_params(self, **parameters):
        self.cls.set_params(**parameters)
        return self
    
##################################################### T E S T S ########################################
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