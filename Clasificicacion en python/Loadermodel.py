# ModelLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


import numpy as np

from sklearn import metrics
from sklearn.metrics._plot import confusion_matrix 

import os
PATHscript = os.path.dirname(__file__)+"/"
os.chdir(PATHscript)


class MLPnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPnet, self).__init__()
        '''
        El método init define las capas de las cuales constará el modelo, 
        aunque no la forma en que se interconectan
        '''
        # Función lineal 1: 784 --> 100
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.tanh1 = nn.Tanh()

        # Función lineal 2: 100 --> 100
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh2 = nn.Tanh()    
        
        # Función lineal 3: (Capa de salida): 100 --> 10
        self.fc4 = nn.Linear(hidden_dim, output_dim)  
        self.sigm4 = nn.ELU()

    def forward(self, x):
        #x = x.view(-1, input_dim)  # aqui convertimos la imagen a un vector unidimensional
        # Capa 1
        z1 = self.fc1(x)
        y1 = self.tanh1(z1)
        # Capa 2
        z2 = self.fc2(y1)
        y2 = self.tanh2(z2)
       
        # Capa 4 (salida)
        z3 = self.fc4(y2)
        out = self.sigm4(z3)
        return out
    
    def name(self):
        return "MLP"



#model1 = MLPnet()
model1 = torch.load("models/modelo600elu")
#model2 = torch.load("")
#model3 = torch.load("")
#model4 = torch.load("")
#clasificadores = []
#clasificadores.append(model1)


# Load data
X_train = []
y_train = []
X_test = []
y_test = []
files = ["Diccionarios/disc_diccAfro.npz", "Diccionarios/disc_diccAsian.npz", "Diccionarios/disc_diccCaucan.npz","Diccionarios/disc_diccLatin.npz"]
for file in files:
    dat1 = np.load(file)
    X_train.append( torch.from_numpy(dat1['xtrain']).to("cuda").float() )
    y_train.append(torch.from_numpy(dat1['ytrain']).to("cuda").float() )
    X_test.append( torch.from_numpy(dat1['xtest']).to("cuda").float() )
    y_test.append( torch.from_numpy(dat1['ytest']).to("cuda").float() ) 

it=0
#for it in range(len(clasificadores)):
    #clasificadores[it].fit(X_train[it],y_train[it])
    #pickle.dump(clasificadores[it], open(listfiles[it], 'wb'))

fig2, axs2 = plt.subplots(2,len(files))
for it in range(len(files)):
    estimacion = model1(X_test[it])
    prediccion = torch.max(estimacion.data, 1)[1] 

    newtest = torch.Tensor.numpy(y_test[it].to('cpu'))
    prenew = 100*torch.Tensor.numpy(prediccion.to('cpu').detach())
    cm = metrics.confusion_matrix(newtest, prenew,normalize='true')
    axs2[0,it] = plt.subplot(2,4,it+1)
    #cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = np.unique(y_test[it]))
    #axs2[0,it]
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['N','H','A'])

#cm = confusion_matrix(model1, X_test[it] , y_test[it], ax= axs[0,it], cmap=plt.cm.Greens,normalize='true')
#
#plt.plot(clasificadores[it].loss_curve_)
    #confusion_matrix = metrics.confusion_matrix()

    
fig2.show()


print("final")