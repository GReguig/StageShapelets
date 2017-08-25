#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:49:38 2017

@author: reguig
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torch.utils.data.sampler import RandomSampler
from torch.utils.data import TensorDataset, DataLoader

from dataFunctions import getData
import numpy as np
import copy
from lossFunction import LogisticLoss


class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
    
    def fit(self, data, target, epochs = 500, batch_size = None, criterion = None, optimizer = torch.optim.Adam, testData = None, testTarget = None, **kwargs):
        self.initNet(data, target)
        opt = optimizer(self.parameters(),weight_decay = self.weight_decay, **kwargs)
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
               test_score = self.score(testData,testTarget)
               self.test_loss.append(test_loss.data[0])
               self.test_score.append(test_score)
               earlyStop = self._earlyStopping(test_score, test_loss)
           else : 
              earlyStop = self._earlyStopping(score, loss)
           #print ("Epoch %d / %d | Loss : %f | Accuracy : %f"%(ep+1, epochs, loss, score))
           #print(list(self.conv.parameters())[0].grad)
           if earlyStop : 
              print("Early stop epoch : %d \n\n"%(ep+1))
              self.load_state_dict(self.best_params)
              break;
        print("Apprentissage fini apres %d epochs"%(ep+1))
        return self
        
    def initNet(self, data, target):
        raise NotImplementedError("Unimplemented network initialisation, please use a subclass of CNN")
        
    def _earlyStopping(self, score, loss, patience = 1000, deltaScore = 0.0001, deltaLoss = 0.001):
        """
        if score - self.best_score > deltaScore :
            self.best_score = score
            self.patience = -1
        """ 
        if score >= self.best_score and np.abs(score - self.best_score) > deltaScore:
            self.best_score = score
            self.patience = -1
            self.best_params = copy.deepcopy(self.state_dict())
            
        self.patience += 1 
        return self.patience > patience 
    
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

##################### Adaptive Stride CNN #######################

class AdaptiveStrideCNN(CNN):
    
    def __init__(self,k = 5, nb_filters = 1,kernel_size = 2, stride = 1, padding = 0, dilation = 1, earlyStopping = True, weight_decay = 0.0, groups = 1, pool = False, kernel_pool = 3):
        super(AdaptiveStrideCNN, self).__init__()
        self.k = k 
        self.earlyStopping = earlyStopping
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.weight_decay = weight_decay
        self.padding = padding if padding else stride
        self.pool = pool
        self.kernel_pool = kernel_pool
    
    def forward(self, data):
        self._adaptConvLayer(data)
        output = data.view(len(data), 1, -1)
        output = F.relu(self.conv(output))
        if self.pool : 
            output = F.max_pool1d(output, self.kernel_pool)
        topk = output.topk(self.k)[0].view(len(data), -1)
        linear = self.linear(topk)
        self._defaultConvParameters
        return F.sigmoid(linear) if self.output_size_== 1 else F.softmax(linear)
    
    def initNet(self, data, target):
        self.conv = nn.Conv1d(1, self.nb_filters, kernel_size = self.kernel_size, stride = self.stride, 
                              padding = self.padding, groups = self.groups)
        #Nombre de classes du dataset
        nbDims = len(target.size())
        nbClasses = len(np.unique(target.data.numpy())) if nbDims == 1 else target.size(1)
        #Nombre de sorties deduites du nombre de classes
        self.output_size_ = nbClasses if nbClasses > 2 else 1
        self.linear = nn.Linear(self.k * self.nb_filters, self.output_size_)
  
    def _adaptConvLayer(self, data):
        #Calcul des dimensions de la sortie
        dimsOutputConv = (data.size(1) + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1
        if self.pool : 
            dimsOutputConv = dimsOutputConv / self.kernel_pool
        #Si les dimensions de sortie ne peuvent pas faire un kmax
        self.newStride, self.newPadding = None, None
        
        if dimsOutputConv < self.k :
            
            #Stride afin d'avoir au moins k valeurs en sortie
            newStride = (data.size(1) + 2 * self.padding - self.dilation * (self.kernel_size - 1))/(self.k - 1) - 1
            
            #Si le stride calcule est non nul, mise a jour du stride de la couche
            if newStride > 0 :
                #print("NewStride : %d"%(newStride))
                self.conv.stride=(newStride, )
                self.newStride = newStride
            else : 
                self.conv.stride = (1, )
                newPadding = (self.k - data.size(1) + self.dilation * (self.kernel_size - 1) + 1) / 2 + 1
                #print("New Padding : %d"%(newPadding))
                self.conv.padding=(newPadding, )
                self.newStride = 1
                self.newPadding = newPadding
                
    def _defaultConvParameters(self):
        self.conv.stride = (self.stride, )
        self.conv.padding = (self.padding, )


    def get_params(self, deep=True):
        return {"k" : self.k, "nb_filters" : self.nb_filters, "kernel_size" : self.kernel_size, "stride" : self.stride,
                "padding" : self.padding, "groups" : self.groups, "weight_decay" : self.weight_decay}
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.padding = self.stride
        return self

########################################## DEEP ADAPTIVE STRIDE CNN ##########################################

class DeepCNNAdaptiveStride(CNN):

    def __init__(self, nbConvLayers = 3, k = 5, kernel_sizes = [2, 2, 2], strides = [1, 1, 1], paddings = [0, 0, 0], weight_decay = 0.0):
        #Verification des arguments
        if len(kernel_sizes) != len(strides) or len(strides) != len(paddings) or len(paddings) != nbConvLayers : 
            raise ValueError("The number of parameters does not match the number of convolutional layers")
        #Initialisation du module
        super(DeepCNNAdaptiveStride, self).__init__()
        self.nbConvLayers = nbConvLayers
        self.k = k
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.weight_decay = weight_decay

    def forward(self, data):
        
        output = data.view(len(data), 1, -1)
        
        for i in range(self.nbConvLayers):
            layer = self.convLayers[i]
            self._adaptConvLayer(output, layer)
            output = F.relu(layer(output))
        self._defaultParameters()
        topk = output.topk(self.k)[0].view(-1, self.k)
        linear = self.linear(topk)
        #activation en sortie du reseau, sigmoid si binaire, softmax sinon
        return F.sigmoid(linear) if self.output_size_== 1 else F.softmax(linear)
        
    
    def initNet(self, data, target):
        self.convLayers = nn.ModuleList([ nn.Conv1d(1, 1, kernel_size = self.kernel_sizes[i], 
                                      stride = self.strides[i], padding = self.paddings[i])
                                      for i in range(self.nbConvLayers) ])
        nbClasses = len(np.unique(target.data.numpy()))
        #Nombre de sorties deduties du nombre de classes
        self.output_size_ = nbClasses if nbClasses > 2 else 1
        #Couche lineaire
        self.linear = nn.Linear(self.k, self.output_size_)
            
    def _adaptConvLayer(self, data, layer):
        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]
        padding = layer.padding[0]
        dilation = layer.dilation[0]
        
        dim_output = (data.size(2) + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        
        if dim_output < self.k : 
            newStride = (data.size(2) + 2 * padding - dilation * (kernel_size - 1))/(self.k - 1) - 1
            if newStride > 0 : 
                layer.stride = (newStride, )
            else : 
                layer.stride = (1, )
                newPadding = np.abs((self.k - data.size(2) + dilation * ( kernel_size - 1) + 1) / 2 + 1)
                layer.padding = (newPadding, )
    
    def _defaultParameters(self):
        for i in range(self.nbConvLayers):
            layer = self.convLayers[i]
            layer.stride = (self.strides[i], )
            layer.padding = (self.paddings[i], )
            
    def get_params(self, deep=True):
        return {"nbConvLayers" : self.nbConvLayers, "kernel_sizes" : self.kernel_sizes, "strides" : self.strides,
                "paddings" : self.paddings, "weight_decay" : self.weight_decay}
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

################################################ MULTIPLE FILTERS CNN ##############


class MultipleFilterSizeCNN(CNN):
    
    def __init__(self, nbConvNets = 3, k = 5, nbFilters = [50, 50, 50], kernel_sizes = [10, 20, 30], strides = [1,1,1], dilations = [1, 1, 1], weight_decay = 0.0):
        super(MultipleFilterSizeCNN, self).__init__()
        self.nbConvNets = nbConvNets
        self.nbFilters = nbFilters
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = strides
        self.dilations = dilations
        self.k = k
        self.weight_decay = weight_decay
        
    def initNet(self, data, target):
        self.convNets = nn.ModuleList([nn.Conv1d(1, out_channels = self.nbFilters[i], kernel_size = self.kernel_sizes[i],
                    stride = self.strides[i], padding = self.paddings[i]) for i in range(self.nbConvNets)])
        nbClasses = len(np.unique(target.data.numpy()))
        #nbClasses = target.size()[1]
        #Nombre de sorties deduties du nombre de classes
        self.output_size_ = nbClasses if nbClasses > 2 else 1
        #Classifieur
        self.linear = nn.Linear(self.k * sum(self.nbFilters), self.output_size_)
    
    def forward(self, data):
        #Donnees de base 
        output = data.view(len(data), 1, -1)
        #Liste des convolutions
        convolutions = []
        for layer in self.convNets:
            self._adaptConvLayer(data, layer)
            convolutions.append(F.relu(layer(output)))
        self._defaultParameters()
        #Liste des kmax par convolution
        kmax = [conv.topk(self.k)[0].view(len(data), -1) for conv in convolutions ]
        kmax = torch.cat(kmax).view(len(data), -1)
        linear = self.linear(kmax)
        return F.sigmoid(linear) if self.output_size_== 1 else F.softmax(linear)
        
        
    def _adaptConvLayer(self, data, layer):
        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]
        padding = layer.padding[0]
        dilation = layer.dilation[0]
        
        dim_output = (data.size(1) + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        
        if dim_output < self.k : 
            newStride = (data.size(1) + 2 * padding - dilation * (kernel_size - 1))/(self.k - 1) - 1
            if newStride > 0 : 
                layer.stride = (newStride, )
            else : 
                layer.stride = (1, )
                newPadding = np.abs((self.k - data.size(1) + dilation * ( kernel_size - 1) + 1) / 2 + 1)
                layer.padding = (newPadding, )
    
    def _defaultParameters(self):
        for i in range(self.nbConvNets):
            layer = self.convNets[i]
            layer.stride = (self.strides[i], )
            layer.padding = (self.paddings[i], )
            
            
"""
l0 = []
l1 = []

for i in range(20) : 
    l0.append(kmax[i]) if trainy.data.numpy()[i] == 0 else l1.append(kmax[i])
    #plt.plot( kmax[i],0, marker = 'o', color='b', label="classe 0") if trainy.data.numpy()[i] == 0 else plt.plot(kmax[i], 0, marker = 'v', color='r', label = "class 1 ")

plt.scatter([i[0] for i in l0], [i[1] for i in l0], marker = 'o', color='b', label='Classe 0')
plt.scatter([i[0] for i in l1], [i[1] for i in l1], marker='v', color='r', label='Classe 1')
x = np.linspace(-10, 10, 100) #* 0.4721) + 0.5016
y = ((x * - 0.8458)/ 0.4721  + 0.5016)
#plt.plot(x,y, label='W')
plt.xlabel("Valeur du 1-max pour le filtre 1")
plt.ylabel("Valeur du 1-max pour le filtre 2")
plt.title("Repartition des series selon leur classe en fonction du 1-max pour deux filtres de taille 30")
plt.legend()
plt.savefig('Discrim2Filtres30ECG.png')
"""            
            
#trainx, trainy, testx, testy = getData("ECG200", oneHot = False)
"""

def toVariable(arr):
    return Variable(torch.Tensor(arr))

"""
#trainx, trainy, testx, testy = toVariable(trainx), toVariable(trainy), toVariable(testx), toVariable(testy)
"""

size = trainx.size(1)

trainx = trainx.view((len(trainx), 1, inputSize))
#1 serie, 1 filtre, noyau de taille 2, stride de 3 pas de temps
conv = nn.Conv1d(1, 1, 2, stride = 3)
linear = nn.Linear(5, 1)
# Traitement du reseau

#Couche de convolution  
convoluted = conv(trainx)
print("Convoluted")
print(convoluted)
#Flatten + 5-max

top5 = convoluted.topk(5)[0].view(len(trainx), -1)
print("Top5")
print(top5)bceloss
#Classifieur
linear = linear(top5)
print("Linear")
print(linear)
#Sortie

output = F.sigmoid(linear)
print("Output")
print(output)

plt.figure()
plt.plot(trainx.data[1].numpy()[0])
plt.plot(c.data[1].numpy()[0])

model = SimpleCNN(kernel_size = 4)
#optimizer = torch.optim.Adagrad(model.parameters(), lr = 0.1, lr_decay = .00001, weight_decay = 0.00 )
model.fit(trainx,trainy, epochs = 10,testData = testx, testTarget = testy, optimizer = torch.optim.Adagrad, lr=0.1, lr_decay = 0.00001, weight_decay = 0.00)




model = AdaptiveStrideCNN(kernel_size = int(0.05 * size), stride = 5, k =int(0.15 * size))
model.fit(trainx, trainy, epochs = 10, lr = 0.1, lr_decay = 0.0001)


for i in range(10):
    plt.figure()
    plt.plot(trainx.data.numpy()[i], label='Serie originale')
    plt.plot(model.convolved.data[i][0].numpy(), label="Serie apres convolution")
    plt.legend()
    plt.title("Serie %d"%(i))
"""
