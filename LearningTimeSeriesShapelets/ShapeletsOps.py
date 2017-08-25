#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 15:14:04 2017

@author: reguig
"""

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from sklearn.cluster import KMeans
import torch.nn.functional as F
import copy

################# CLASSIFIEUR LOGISTIQUE #############################    

class LogisticShapeletsLearner(nn.Module):

    def __init__(self, nb_shapelets = 15, shapelets_size = 10, scale = 3 , alpha = -30, weight_decay = 0.0):
        super(LogisticShapeletsLearner, self).__init__()
        self.nb_shapelets = nb_shapelets
        self.shapelets_size = shapelets_size
        self.scale = scale
        self.alpha = alpha
        self.weight_decay = weight_decay
    
    def initNet(self, series, targets):
        nbDims = len(targets.size())
        nbClasses = len(np.unique(targets.data.numpy())) if nbDims == 1 else targets.size(1)
        
        self.output_size_ = targets.size(1) if len(targets.size()) > 1 else 1
        self.shapelets = seriesToShapeletsVariousLength(series.data.numpy(), self.shapelets_size, self.nb_shapelets, self.scale)
        self.linear = nn.Linear(self.scale * self.nb_shapelets, nbClasses)
    
    def forward(self, series):
        softmin = _softMinShapeletsSeriesVariousLength(self.shapelets, series, self.alpha)
        out = self.linear(softmin)
        return F.sigmoid(out) if self.output_size_ == 1 else F.softmax(out)
    
    def fit(self, data, target, epochs = 500, batch_size = None, criterion = None, optimizer = torch.optim.Adam, testData = None, testTarget = None, **kwargs):
        self.initNet(data, target)
        opt = optimizer(self.parameters(), weight_decay = self.weight_decay, **kwargs)
        self.opt = opt
        batch = batch_size != None
        if batch : 
            steps = np.ceil(1.0*len(data)/batch_size)
            lr = opt.param_groups[0]['lr']
            opt.param_groups[0]['lr'] = 1.0 * lr / steps 
        indexes = range(len(data))
        test = testData and testTarget
        nbClasses = len(np.unique(target.data.numpy()))
        toLong = False
        if not criterion:
            #criterion = nn.BCELoss() if nbClasses == 2 else nn.CrossEntropyLoss()
            criterion = nn.MSELoss()
            #toLong = nbClasses == 2
        #Historique du loss/score
        self.histo_loss_ = []
        self.score_histo = []
        self.test_loss = []
        self.test_score = []
        self.best_score = 0.0
        self.best_loss = float("inf")
        self.patience = 0
        for ep in range(epochs):
           #Train avec mini batch
         
           if batch : 
               #Melange des indices
               np.random.shuffle(indexes)
               
               tmpLoss = 0.0
               for i in range(1,int(steps)+1):
                   #Remise a zero du gradient
                   opt.zero_grad()
                   #Indice du premier exemple
                   indexDepart = batch_size * (i-1)
                   #Indice du derneir exemple
                   indexStop = batch_size * (i) if batch_size * (i-1) < len(data) - 1 else  None
                   #Exemples du batches
                   dataBatch = data[torch.Tensor(indexes[indexDepart:indexStop]).long()]
                   targetBatch = target[torch.Tensor(indexes[indexDepart:indexStop]).long()]
                   #Calcul de la sortie du reseau
                   output = self(dataBatch)
                   #Calcul du loss
                   loss = criterion(output, targetBatch ) #if toLong else criterion(output, targetBatch.long())
                   tmpLoss += loss.data[0]
                   loss.backward()
                   opt.step()
               self.histo_loss_.append(tmpLoss)
               loss = tmpLoss
               score = self.score(data, target)
               self.score_histo.append(score)
  
           else :         
               opt.zero_grad()
               output = self(data)
               loss = criterion(output, target) #if toLong else criterion(output, target.long())
               self.histo_loss_.append(loss.data[0])
               score = self.score(data,target)
               self.score_histo.append(score)
               loss.backward()
               opt.step()
               loss = loss.data[0]
           earlyStop = False
           if test : 
               test_loss = criterion(self(testData), testTarget) #if toLong else criterion(self(testData), testTarget)
               test_score = self.score(testData,testTarget).data[0]
               self.test_loss.append(test_loss.data[0])
               self.test_score.append(test_score)
               earlyStop = self._earlyStopping(test_score, test_loss)
           else : 
              earlyStop = self._earlyStopping(score, loss)
           print("Epoch %d/%d | Loss : %f | Accuracy :  %f"%(ep+1, epochs, loss, score))
           if earlyStop : 
              print("Early stop epoch : %d \n\n"%(ep+1))
              self.load_state_dict(self.best_params)
              break;
        print("Apprentissage fini apres %d epochs"%(ep+1))
    
    def predict(self, data):
        if self.output_size_ > 1 :
            _, argmax = torch.max(self(data), 1)
            """
            res = torch.zeros(len(data),self.output_size_)
            argmax = argmax.view(-1).data.long()
            for i in range(len(argmax)):
                res[i, argmax[i]] = 1
            return res
            """
            return argmax
        return (self(data) > 0.5).float()
    
    def score(self, series, targets):
        if len(targets.size())== 1 : 
            return (targets.data.float() == self.predict(series).data.float()).float().mean()
        
        argmax = self.predict(series)
        _, argTargs = targets.max(dim=-1)
        
        return (argmax.data.float() == argTargs.data.float()).float().mean()
        #return (targets.data.long() == self.predict(series).data.long()).float().mean()
    
    def clone(self):
        return copy.deepcopy(self)

    def _earlyStopping(self, score, loss, patience = 1000, deltaScore = 0.001, deltaLoss = 0.001):
        """
        if score - self.best_score > deltaScore :
            self.best_score = score
            self.patience = -1
        """ 
        if loss < self.best_loss and np.abs(loss - self.best_loss) > deltaLoss:
            self.best_loss = loss
            self.patience = -1
            self.best_params = copy.deepcopy(self.state_dict())
            
        self.patience += 1 
        return self.patience > patience 
    
    #def __init__(self, nb_shapelets = 15, shapelets_size = 10, scale = 3 , alpha = -30):
    def get_params(self, deep=True):
        return {"nb_shapelets" : self.nb_shapelets, "shapelets_size" : self.shapelets_size, 
                "scale" : self.scale, "alpha" : self.alpha, "weight_decay" : self.weight_decay}
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

################################################### FONCTIONS SUR SHAPELETS ##################


def serieToShapelets(serie, shapelet_size):
    tailleSerie = len(serie)
    indiceFinal = tailleSerie - shapelet_size + 1
    return np.array([serie[indiceDepart : indiceDepart + shapelet_size] for indiceDepart in range(indiceFinal)])
    
def seriesToShapelets(series, shapelet_size):
    return np.array([serieToShapelets(serie, shapelet_size) for serie in series]).reshape(-1, shapelet_size)

def getCentroids(series, shapelet_size, nb_shapelets):
    seriesAsShapelets = seriesToShapelets(series, shapelet_size)
    return KMeans(n_clusters = nb_shapelets, n_jobs = 1).fit(seriesAsShapelets).cluster_centers_

def seriesToShapeletsVariousLength(series, minSize, nb_shapelets, scale = 1):
    #Liste de toutes les tailles de shapelets a generer
    tailleShapelets = [s * minSize for s in range(1, scale+1)]
    #Matrice contenant tous les shapelets, les  variations de tailles sont gerees par padding avec des 0
    shapelets = [Variable(torch.from_numpy(getCentroids(series, size, nb_shapelets))) for size in tailleShapelets]
    """
    shapelets = np.zeros((len(tailleShapelets) * nb_shapelets, tailleShapelets[-1]))
    #Colonne a laquelle on est actuellement
    indexCol = 0
    #Parcours de toutes les tailles de shapelets possibles
    for size in tailleShapelets:
        #Generation des valeurs initiales des shapelets (centroides des sous-sequences de la serie)
        shapelets[indexCol : indexCol+nb_shapelets,:size] = getCentroids(series, size, nb_shapelets)
        #Incrementation de l'indice de colonne
        indexCol += nb_shapelets
    """ 
    return shapelets
    

##################################################### CALCUL DU SOFTMIN ###########################

def _distanceShapelets(shapelet1, shapelet2):
    """
    Permet de calculer la distance entre deux shapelets de meme taille 
        Arguments : 
            :param shapelet1: tenseur contenant un unique shapelet
            :param shapelet2: tenseur contenant les shapelets avec lesquels on calcule la distance
        Retourne :
            :return: tenseur contenant la distance entre les shapelets
    """
    #return (shapelet1 - shapelet2).pow(2).mean(dim=1)
    return (shapelet1.expand(shapelet2.size()) - shapelet2).pow(2).mean(dim=1)


def _distanceShapeletSeries(shapelet, series):
    """
    Permet de calculer la distance entre un shapelet et un ensemble de series
        Arguments :
            :param shapelet: shapelet avec lequel on calcule les distances
            :param series: tenseur contenant l'ensemble des series avec lesquelles on calcule les distances
        Retourne : 
            :return: tenseur contenant pour chaque pas de temps possible la distance entre le shapelet et la sous-sequence de la serie
    """

    tailleShapelet = shapelet.size(1)
    tailleSerie = series.size()[1]
    indiceFinal = tailleSerie - tailleShapelet + 1 
    return torch.cat ( [_distanceShapelets(shapelet, series.narrow( 1, indiceDepart, tailleShapelet ).float() )
                                    for indiceDepart in range(indiceFinal) ], dim = 1)



def _softMinShapeletSeries(shapelet, series, alpha = -30):
    """
    Permet de calculer le softMin de la distance
        Arguments : 
            :param tensorDistances: tenseur contenant les distances d'une serie par rapport a un shapelet (cf fonction '_distanceShapeletSerie')
        Retourne : 
            :return: tenseur contenant la distance minimale approchee (softmin)
    """
    tensorDistances = _distanceShapeletSeries(shapelet, series)
    expDist = torch.exp(alpha * tensorDistances) + 0.0001 #Biais pour eviter division par 0
    numerateur = (tensorDistances * expDist).sum(dim=1)
    denominateur = expDist.sum(dim=1)
    return numerateur / denominateur.expand(numerateur.size())
    

def _softMinShapeletsSeries( shapelets, series, alpha = -30):
    """
    Permet de calculer la representation softmin des series en fonction des shapelets
        Arguments : 
            :param shapelets: tenseur de shapelets
            :param series: tenseur de series
        Retourne :
            :return: tenseur de representation des series en fonction de la distance par rapport aux shapelets
    """
    return torch.cat([_softMinShapeletSeries(s.view(1,-1), series, alpha = alpha) for s in shapelets ], dim = 1)

def _softMinShapeletsSeriesVariousLength(shapelets, series, alpha = -30):
    """
    Permet de calculer la representation softmin des series en fonction d'une liste de tenseurs de shapelets
        Arguments : 
            :param shapelets: liste de tenseurs contenant des shapelets
            :param series : tenseur de series
        Retourne:
            :return: tenseur de representation des series en fonction des distances par rapport aux shapelets
    """
    softMins = [_softMinShapeletsSeries(shapeletsScale, series, alpha = alpha) for shapeletsScale in shapelets ]
    return torch.cat(softMins, dim = 1)




"""
plt.ion()
plt.figure()


a = np.array([1, 2, 3])

plt.plot(a)

for i in range(200):
    a = a+1
    plt.draw()
    plt.pause(0.05)
    

while(True):
    plt.pause(0.05)

"""

"""
########################################### T E S T S #########################################
#donnees
path = r"/home/reguig/datasets/UCR/Data/Sony"
#Donnees
data = np.load(path+"/X.npy")
#Labels
target = np.load(path+"/Y.npy")

#Vecteur de poids
#Shapelet (cense etre initialise autrement)
series = Variable(torch.from_numpy(data[:100])).float()
shapelets = torch.from_numpy(data[:3,:5]).float()
classes = Variable(torch.from_numpy(np.array([target[:100]]))).float()

#Modele
net = LearningShapeletsLogistic(shapelets, 2, alpha = -100)
#Fonction de cout
#criterion = nn.BCELoss()
criterion = nn.MSELoss() 
#Algorithme d'optimisation
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
l = []
plt.figure()
for k in range(500):
    #Mise a Zero du gradient   
    #for i in range(net.shapelets.size()[0]):
    #    plt.plot(net.shapelets[i].data.numpy(), label="Shapelet %d"%(i))
    if k % 50 == 0 :
        plt.plot(net.shapelets[1].data.numpy(), label="Epoch %d"%(k))
    optimizer.zero_grad()
    output = net(series)
    loss = criterion(output, classes)
    loss.backward()
    optimizer.step()
    l.append(loss)
plt.legend()
plt.title("Evolution d'un shapelet")
l = torch.cat(l)
"""