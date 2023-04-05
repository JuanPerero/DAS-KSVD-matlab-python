import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from importMATexperiment import matset2np


###################################################################################################
## Secuencia para el entrenamiento con los diccionatios de matlab
# 1 - abrir:
#       - archivo de diccionario .mat
#       - archivo de Folds.mat (combinacion de segmentos para los conjuntos)
#       - archivo data.mat (señales y etiquetas en formato de matriz)
#       - archivo indxfolds.mat (indices de las señales del los folds)
# 2 - Un conjunto de train/test se conforma de la forma
#     -  folds --> idxfolds --> data
# 3 - Utilizar el archivo "importMATexperiment.py"
###################################################################################################


# Custom Dataset Class
## Needs ATLEAST 3 class methods
## __init__, __len__, __getitem__

class CustomStarDataset(Dataset):
    # This loads the data and converts it, make data rdy
    def __init__(self,xdata,ydata):       
        # conver to torch dtypes
        self.dataset=xdata.float()
        self.labels=ydata.long()
    
    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.dataset)
    
    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        return self.dataset[idx],self.labels[idx]







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



#from scipy.io import loadmat
import numpy as np
from spams import omp
'''
file = np.load("disc_diccAfro.npz")
X_train = file['xtrain']
y_train = file['ytrain']
X_test = file['xtest']
y_test = file['ytest']
'''

importer = matset2np()
pathfiles  ="/home/jperero/Escritorio/Super proyecto/proyecto Matlab/das-ksvd-main/"
dicfile  = pathfiles +"disc_diccCaucadic1.mat" 
datfile  = pathfiles +"disc_diccCauca-data.mat"
idxfile  = pathfiles +"disc_diccCauca-indxfolds.mat"
foldsfile= pathfiles+"disc_diccCauca-folds.mat"
importer.compilardatos(dicfile, datfile, idxfile, foldsfile)
fullX_train, y_train, fullX_test, y_test = importer.getfold(4)

nonzero = fullX_train.shape[0]*0.2 

X_train = omp(fullX_train,importer.diccionario,L=nonzero).toarray().T
X_test = omp(fullX_test,importer.diccionario,L=nonzero).toarray().T








y_train = y_train/200
y_test = y_test/200

X_train = torch.from_numpy(X_train).to("cuda").float()
y_train = torch.from_numpy(y_train).to("cuda")
X_test = torch.from_numpy(X_test).to("cuda").float()
y_test = torch.from_numpy(y_test).to("cuda")

train = CustomStarDataset(X_train,y_train)
test =  CustomStarDataset(X_test,y_test)

# X_train = X_train.type(torch.float32)
#y_train = y_train.type(torch.float32)
#X_test = X_test.type(torch.float32)
#y_test = y_test.type(torch.float32)

#batch_size = 200
#n_iters    = 1000
#num_epochs = int( n_iters / (y_train.size()[0] / batch_size) )

#print("Cantidad de datos: ", y_train.size())
#print("Tamaño de los batchs: ", batch_size)
#print("Cantidad de Iteraciones: ", n_iters)
#print("Numero de epocas: ", num_epochs)

batch_size = 200
num_epochs = 100

#train= torch.cat((X_train,y_train.view(-1,1)),1) 
#test = torch.cat((X_test,y_test.view(-1,1)),1) 


train_loader = torch.utils.data.DataLoader(dataset    = train,
                                           batch_size = batch_size,
                                           shuffle    = True)
test_loader  = torch.utils.data.DataLoader(dataset    = test,
                                           batch_size = batch_size,
                                           shuffle    = False)



input_dim  = 150
hidden_dim = 600    # número de neuronas en las capas ocultas
output_dim = 3    # número de etiquetas
learning_rate = 0.01

model = MLPnet(input_dim, hidden_dim, output_dim)
use_cuda = torch.cuda.is_available()


if use_cuda:
    model = model.cuda()

#Crear instancia de la función de pérdida
error = nn.CrossEntropyLoss()
#Crear instancia del Optimizador
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)






#######################################################################################
####  ---------------------------- ENTRENAMIENTO --------------------------------  ####
#######################################################################################

loss_list         = []
iteration_list    = []
accuracy_list     = []
accuracy_list_val = []

for epoch in range(num_epochs):
    total=0
    correct=0
    # - - - - - - - - - - - - - - - 
    # Entrena la Red en lotes cada época
    # - - - - - - - - - - - - - - - 
    for i, (images, labels) in enumerate(train_loader):
        
        if use_cuda:                              # Define variables
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images) 
        labels = Variable(labels)
        
        optimizer.zero_grad()                      # Borra gradiente
        outputs = model(images)                    # Propagación
        loss    = error(outputs, labels)           # Calcula error
        loss.backward()                            # Retropropaga error
        optimizer.step()                           # Actualiza parámetros
        
        predicted = torch.max(outputs.data, 1)[1]  # etiqueta predicha (WTA)
        total += len(labels)                       # número total de etiquetas en lote
        correct += (predicted == labels).sum()     # número de predicciones correctas
        
    # calcula el desempeño en entrenamiento: Precisión (accuracy)
    accuracy = float(correct) / float(total)
    # almacena la evaluación de desempeño
    iteration_list.append(epoch)
    loss_list.append(loss.item())
    accuracy_list.append(accuracy)

    # - - - - - - - - - - - - - - - 
    # Evalúa la predicción en lotes cada época
    # - - - - - - - - - - - - - - - 
    correct = 0
    total   = 0
    for images, labels in test_loader: 


        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images)                   # Define variables
        labels = Variable(labels)
        
        outputs = model(images)                     # inferencia

        predicted = torch.max(outputs.data, 1)[1]   # etiqueta predicha (WTA)  
        total += len(labels)                        # número total de etiquetas en lote
        correct += (predicted == labels).sum()      # número de predicciones correctas

    # calcula el desempeño: Precisión (accuracy)
    accuracy_val = float(correct) / float(total)
    accuracy_list_val.append(accuracy_val)

    # - - - - - - - - - - - - - - - 
    # Despliega evaluación
    # - - - - - - - - - - - - - - - 
    print('Epoch: {:02}  Loss: {:.6f}  Accuracy: {:.6f}  Accuracy Val: {:.6f}'.format(epoch, loss.data, accuracy, accuracy_val))    


# Loss
fig, axs = plt.subplots(1,1)
plt.plot(iteration_list,loss_list)
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Loss Train")
fig.show()

# Accuracy
fig2, axs2 = plt.subplots(1,1)
plt.plot(iteration_list,accuracy_list,'b')
plt.plot(iteration_list,accuracy_list_val, 'g')
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.title("Accuracy: Train - Val ")
fig2.show()

torch.save(model, 'modelo600elu')
print("FINISHED")



#model = TheModelClass(*args, **kwargs)
#model.load_state_dict(torch.load(PATH))
#model.eval()