# Model Loader
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

import os
PATHscript = os.path.dirname(__file__)+"/"
os.chdir(PATHscript)




model1 = torch.load("models/modelo600elu")
#model2 = torch.load("")
#model3 = torch.load("")
#model4 = torch.load("")
clasificadores = []
clasificadores.append(model1)


# Load data
X_train = []
y_train = []
X_test = []
y_test = []
files = ["Diccionarios/disc_diccAfro.mat", "Diccionarios/disc_diccAsian.mat", "Diccionarios/disc_diccCaucan.mat","Diccionarios/disc_diccLatin.mat"]
for file in files:
    dat1 = np.load(file)
    X_train.append(['xtrain'])
    y_train.append(['ytrain'])
    X_test.append(['xtest'])
    y_test.append(['ytest'])


for it in range(len(clasificadores)):
    #clasificadores[it].fit(X_train[it],y_train[it])
    #pickle.dump(clasificadores[it], open(listfiles[it], 'wb'))

   
 
    prediccion = clasificadores[it].(X_test[it])
    
    cm = plot_confusion_matrix(clasificadores[it], X_test[it] , y_test[it], ax= axs[0,it], cmap=plt.cm.Greens,normalize='true')
    axs[1,it] = plt.subplot(2,4,it+len(clasificadores)+1)
    plt.plot(clasificadores[it].loss_curve_)
    #confusion_matrix = metrics.confusion_matrix()

    
fig.show()

fig2, axs2 = plt.subplots(4,4)
for it in range(len(clasificadores)):
    for j in range(len(clasificadores)):
        cm = plot_confusion_matrix(clasificadores[it], X_test[j] , y_test[j], ax= axs2[j,it], cmap=plt.cm.Greens,normalize='true')
        
fig2.show()
print("final")