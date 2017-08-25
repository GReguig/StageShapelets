#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:36:31 2017

@author: reguig
"""
import numpy as np
import h5py
import os
import torch
from torch.autograd import Variable

path = "/home/reguig/datasets/UCR"

def getData(datasetName, oneHot = False):
    p = path+"/"+datasetName if "/" not in datasetName else datasetName
    #trainx = Variable(torch.Tensor(np.load(p + "/X_Train.npy")))
    #testx = Variable(torch.Tensor(np.load(p + "/X_Test.npy")))
    trainx = np.load(p + "/X_Train.npy")
    testx = np.load(p + "/X_Test.npy")
    if oneHot : 
        
        #trainy = Variable(torch.Tensor(ToOneHot(np.load(p + "/Y_Train.npy"))))
        #testy = Variable(torch.Tensor(ToOneHot(np.load(p + "/Y_Test.npy"))))
        trainy = ToOneHot(np.load(p + "/Y_Train.npy"))
        testy = ToOneHot(np.load(p + "/Y_Test.npy"))
    else : 
        #trainy = Variable(torch.Tensor(np.load(p + "/Y_Train.npy")))
        #testy = Variable(torch.Tensor(np.load(p + "/Y_Test.npy")))
        trainy = np.load(p + "/Y_Train.npy")
        testy = np.load(p + "/Y_Test.npy")
    
    return trainx, trainy, testx, testy

"""
Permet d'extraire les datasets d'un fichier h5 et de les sauvegarder au format npy
"""
def unpackH5(fichier, destination=""):
    #Ouverture en ecriture
    f = h5py.File(fichier, 'r')
    #f['yoga']['TEST']['labels']['values'].value
    for datasetName in f.keys():
        print("Dataset : %s"%(datasetName))
        X_Train = f[datasetName]['TRAIN']['ts']['block0_values'].value.squeeze()
        Y_Train = f[datasetName]['TRAIN']['labels']['values'].value 
        X_Test = f[datasetName]['TEST']['ts']['block0_values'].value.squeeze()
        Y_Test = f[datasetName]['TEST']['labels']['values'].value 
        replaceLabels(Y_Train)
        replaceLabels(Y_Test)
        pathdir = destination + "/" + datasetName
        if not os.path.exists(pathdir):
            os.mkdir(pathdir)
        print("Dossier %s cree"%(pathdir))
        np.save(pathdir+"/X_Train", X_Train)
        np.save(pathdir+"/Y_Train", Y_Train)
        np.save(pathdir+"/X_Test", X_Test)
        np.save(pathdir+"/Y_Test", Y_Test)
        print("Donnees sauvegardees au format npy")
        
    f.close()    

"""
Permet de transformer un vecteur de labels en un ensemble de vecteurs oneHot
    Argument
        Y : array-like des labels
    Retourne
        Un ensemble de vecteurs de forme oneHot
"""

def ToOneHot(Y):
    #Liste des différentes classes
    classes = np.unique(Y)
    #Reshape des exemples, traitement des liste en array
    Y = np.asarray(Y).reshape((-1,1))
    #Nouvelle matrice des labels codées en onehot
    onehot = np.zeros((len(Y),len(classes)))
    
    for i in range(len(classes)):
        #Liste des index des exemples de classe classes[i]
        tmp = np.where(Y==classes[i])[0]
        onehot[tmp,i] = 1
    return onehot


def replaceLabels(labels):
    classes = np.unique(labels)
    copie = labels.copy()
    nbClasses = len(classes)
    listClasses = range(nbClasses)
    for i in listClasses : 
        classe = classes[i]
        labels[np.where(copie == classe)] = i 
        
        