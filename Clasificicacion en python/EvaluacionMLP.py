import os
PATHscript = os.path.dirname(__file__)+"/"
os.chdir(PATHscript)

import numpy as np
from scipy.io import loadmat
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import pickle
import matplotlib.pyplot as plt
#from sklearn.metrics import plot_confusion_matrix


#Clasificador MLP
hidenlayers = (500)
funcactivacion = 'tanh' #'tanh' ‘identity’, ‘logistic’, ‘relu’
solver = 'sgd'  # 'sgd' 'adam' 
learning_rate = 0.05
maxitermlp = 1000
tolerancia = 0.000001

file1 = loadmat("disc_diccAfro.mat")
file2 = loadmat("disc_diccAsian.mat")
file3 = loadmat("disc_diccCaucan.mat")
file4 = loadmat("disc_diccLatin.mat")

file5 = loadmat("resultSHHS.mat")

files = [file1, file2, file3, file4,file5]

X_train = []
y_train = []
X_test = []
y_test = []
clasificadores = []

for file in files:
    X_train.append(np.array(file['data_train']).T)
    y_train.append(np.array(file['labels_train'])[0])
    X_test.append(np.array(file['data_test']).T)
    y_test.append(np.array(file['labels_test'])[0])
    clf = MLPClassifier(hidden_layer_sizes = hidenlayers,      # arquitectura de la red
                        activation = funcactivacion,             # función de activación
                        solver = solver,                 # algoritmo de optimización
                        learning_rate_init = learning_rate,      # tasa de aprendizaje
                        max_iter = maxitermlp,                  # épocas de entrenamiento
                        tol = tolerancia,                 # tolerancia de error
                        verbose = False)      
    clasificadores.append(clf)

listfiles = ["modeloDASmultdicc1.model", "modeloDASmultdicc2.model", "modeloDASmultdicc3.model", "modeloDASmultdicc4.model", "modeloSHHS.model"]                       

fig, axs = plt.subplots(2,len(clasificadores))
for it in range(len(clasificadores)):

    if(os.path.isfile(listfiles[it])):
        clasificadores[it] = pickle.load(open(listfiles[it], "rb"))
    else:
        clasificadores[it].fit(X_train[it],y_train[it])
        pickle.dump(clasificadores[it], open(listfiles[it], 'wb'))

    prediccion = clasificadores[it].predict(X_test[it])
    
    cm = metrics.confusion_matrix(y_test[it], prediccion,normalize='true')
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['N','A','H']).plot(ax = axs[0,it],cmap=plt.cm.Greens,colorbar=False)
    # Metodo con sklearn viejo por plot_confusion
    #cm = plot_confusion_matrix(clasificadores[it], X_test[it] , y_test[it], ax= axs[0,it], cmap=plt.cm.Greens,normalize='true')
    
    axs[1,it] = plt.subplot(2,len(clasificadores),it+len(clasificadores)+1)
    plt.plot(clasificadores[it].loss_curve_)
  
fig.show()

fig2, axs2 = plt.subplots(len(files),len(clasificadores))
for it in range(len(clasificadores)):
    for j in range(len(files)):
        predic = clasificadores[it].predict(X_test[j])
        cm = metrics.confusion_matrix(y_test[j], predic,normalize='true')
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm).plot(ax = axs2[j,it],cmap=plt.cm.Greens,colorbar=False)
        #cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['N','A','H']).plot(ax = axs2[j,it],cmap=plt.cm.Greens,colorbar=False)
fig2.show()
print("final")
