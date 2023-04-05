from scipy.io import loadmat
import numpy



class matset2np:
    def __init__(self):
        self.diccionario =None
        self.data = None
        self.idxfolds = None
        self.folds = None
        return

    def convdicc(self,diccfile):
        self.diccionario = loadmat(diccfile)
        self.diccionario = self.diccionario['Phi_D'][0,0]
        return self.diccionario

    def convdata(self,partialfile):
        self.data = loadmat(partialfile)
        self.data = [self.data['signals'],self.data['targets'][0]]
        return self.data

    def convfold(self,idxfold,foldsfile):
        convfolds = loadmat(foldsfile)
        signalsfolds = loadmat(idxfold)

        ftest = signalsfolds['inxfoldstest']
        ftrain = signalsfolds['inxfoldstrain']
        sigindx = convfolds['folds']
        self.folds = [ftrain-1, ftest-1]
        self.idxfolds = sigindx-1
        return sigindx, ftrain, ftest

    def prossesdata(self, datfile, idxfile, foldsfile, check=False, checksize=(128,150)):
        ret = self.convdata(datfile)
        if len(ret[0].shape)!=2 or ret[0].shape[0]!=checksize[0] or ret[0].shape[1]!=ret[1].shape[0]:
            raise ValueError("ERROR EN EL TAMAÑO DE LA MATRIZ DE DATOS")      
        orden = self.convfold(idxfile, foldsfile)
        return


    def compilardatos(self, dicfile, datfile, idxfile, foldsfile, check=False, checksize=(128,150)):
        dic = self.convdicc(dicfile)
        if dic.shape != checksize:
            raise ValueError("ERROR EN EL TAMAÑO DEL DICCIONARIO")
        ret = self.convdata(datfile)
        if len(ret[0].shape)!=2 or ret[0].shape[0]!=checksize[0] or ret[0].shape[1]!=ret[1].shape[0]:
            raise ValueError("ERROR EN EL TAMAÑO DE LA MATRIZ DE DATOS")      
        orden = self.convfold(idxfile, foldsfile)
        return

    def getfold(self,num):
        ftrain = self.folds[0][num] #train
        ftest  = self.folds[1][num] #test

        xtrain = self.data[0][:,self.idxfolds[ftrain].reshape(1,-1)[0]]
        ytrain = self.data[1][self.idxfolds[ftrain].reshape(1,-1)[0]]
        xtest  = self.data[0][:,self.idxfolds[ftest].reshape(1,-1)[0]]
        ytest  = self.data[1][self.idxfolds[ftest].reshape(1,-1)[0]]
        return xtrain, ytrain, xtest, ytest
    






