#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 10:25:31 2017

@author: reguig
"""
from ShapeletsOps import ShapeletsLearner
import torch
import torch.nn as nn
from torch.autograd import Variable

from lossFunction import LogisticLoss

"""
Modele lineaire avec 1 couche cachee
"""
class LogisticRegShapelets(ShapeletsLearner):
    
    def __init__(self, **kwargs):
        super(LogisticRegShapelets, self).__init__(**kwargs)
        self.linear = nn.Linear(self.nbShapelets, self.output_size)
        self.shapelets_histo = []
        
    def forward(self, series):
        #Conversion de la serie en representation en fonction des distances par rapport aux shapelets
        projection = self._softMinShapeletsSeries(self.shapelets, series)
        return self.linear(projection)
    
    def fit(self, series, targets, criterion = LogisticLoss, optimizer = None, lr = 0.001, epochs = 500):
        optimizer = torch.optim.SGD(self.parameters(), lr = lr, weight_decay = 0.0) if not optimizer else optimizer
        for ep in range(epochs) : 
            print("epoch : %d"%(ep))
            if ep % 10 == 0 : 
                self.shapelets_histo.append(self.shapelets.data.numpy().copy())
            optimizer.zero_grad()
            outputs = self(series)
            loss = criterion.forward(outputs, targets)     
            print("Loss ")
            print(loss.data[0])
            self.loss_histo.append(loss.data[0])
            loss.backward()
            optimizer.step()
            
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
    

"""
trainx, trainy, testx, testy = getData("ItalyPowerDemand")    
#trainy, testy = trainy.long(), testy.long()
#testy[testy == 0] = -1
nbShapelets = int(0.3 * trainx.size(1))
taille = int(0.2 * trainx.size(1))

#net = LogisticRegShapelets(nb_shapelets = 2, tailleShapelets = 5, series = trainx, nb_classes = 2, alpha = -30)
net = CNNShapelets(nb_shapelets = nbShapelets, tailleShapelets = taille, series = trainx, nb_classes = 2, scale=3, alpha=-30)
#criterion = nn.BCELoss()
#criterion = nn.MSELoss()
criterion = LogisticLoss()
#criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD([
        {'params':[net.shapelets], 'lr': 0.001},
        {'params':net.linear.parameters(), 'lr':0.001, 'weight_decay' : 0.01}
        ])
    
#optimizer = torch.optim.Adagrad(net.parameters(), lr = 0.1, lr_decay = .00001, weight_decay = 0.001 )

optimizer = torch.optim.Adagrad([
        {'params':[net.shapelets], 'lr': 0.01, 'lr_decay':.0000},
        {'params':list(net.parameters())[1:], 'lr':0.01, 'lr_decay' : .0000, 'weight_decay' : 0.01}
        ])
net.fit(trainx, trainy, criterion, optimizer = optimizer, epochs = 5000, dataTest = testx, targetTest = testy)

net.score(testx,testy)

for i in range(5) : 
    plt.plot(net.shapelets[-i].data.numpy(), label='shapelet %d'%(i))
plt.legend()

for epoch in range(len(net.shapelets_histo)) : 
    
    for i in range(3):
        plt.figure(i)
        plt.plot(net.shapelets_histo[epoch][i*2+4], label = "epoch%d"%(epoch*10))
    for i in range(3):
        plt.figure(i)
        plt.legend()
sh1 = net.shapelets[0]
sh2 = net.shapelets[1]
sh12 = Variable(torch.from_numpy(net.shapelets_histo[0][0]))
sh22 = Variable(torch.from_numpy(net.shapelets_histo[0][1]))

projSh1 = net._softMinShapeletSeries(sh1, trainx).data.numpy()
projSh2 = net._softMinShapeletSeries(sh2, trainx).data.numpy()

projSh12 = net._softMinShapeletSeries(sh12, trainx).data.numpy()
projSh22 = net._softMinShapeletSeries(sh22, trainx).data.numpy()
#colors = ['+' if target == 0 else '^' for target in trainy.data.numpy()]

targetnp =  trainy.data.numpy()
plt.figure()
plt.scatter(projSh1[targetnp==0], projSh2[targetnp==0],  marker = "^", c='red')
plt.scatter(projSh1[targetnp==1], projSh2[targetnp==1], marker = "x", c='blue')
#plt.plot(next(net.parameters())[0].data.numpy())
plt.title("Epoch 100")

plt.figure()
plt.scatter(projSh12[targetnp==0], projSh22[targetnp==0],  marker = "^", c='red')
plt.scatter(projSh12[targetnp==1], projSh22[targetnp==1], marker = "x", c='blue')
#plt.plot(next(net.parameters())[0].data.numpy())
plt.title("Epoch 10")
data = np.hstack((projSh1, projSh2))
data2 = np.hstack((projSh12, projSh22))
def make_grid(data, step=20):
    xmax, xmin, ymax, ymin = np.max(data[:,0]),  np.min(data[:,0]), np.max(data[:,1]), np.min(data[:,1])
    x, y =np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step), np.arange(ymin,ymax,(ymax-ymin)*1./step))
    grid=np.c_[x.ravel(),y.ravel()]
    return grid, x, y

plt.figure()
grid, x, y = make_grid(data, step=200)
vargrid = Variable(torch.from_numpy(grid)).float()
plt.contourf(x, y, net._predict(vargrid.view(len(vargrid), 1, vargrid.size(1))).data.numpy().reshape(x.shape), colors=('gray', 'orange'), levels=[-1, 0, 1])
plt.scatter(projSh1[targetnp==0], projSh2[targetnp==0],  marker = "^", c='red')
plt.scatter(projSh1[targetnp==1], projSh2[targetnp==1], marker = "x", c='blue')
#plt.scatter(projSh12[targetnp==0], projSh22[targetnp==0],  marker = "^", c='red')
#plt.scatter(projSh12[targetnp==1], projSh22[targetnp==1], marker = "x", c='blue')

plt.plot(net.loss_histo)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss en train en fonction de l'epcoh pour le dataset ItalyPowerDemand")
plt.savefig("ItalyPowerDemandTrain")

"""