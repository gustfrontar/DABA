#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 19:25:40 2024

@author: jruiz
"""

import numpy as np
import torch
from torch.utils.data import Dataset

#Funcion para abrir los datos .npz y extraer las variables que elegimos, y lo guardamos en un diccionario
def get_data( Input , Target , split_ratios ) :
    #Input array nvar x ninstances
    #Target array nvar x ninstances (same shape as Input)
    #Split ratios ( train_data_ratio , val_data_ratio ) sum of Split ratios is equal to one so test_data_ratio is obtained from the other two
    
    
    Data = dict()
    #En este caso tanto el input como el target tiene las mismas dimensiones, por eso podemos usarlas para ambos
    Data["len_total"] , Data["nx"]  = Input.shape
    
    train_si = 0 
    train_ei = int( split_ratios[0] * Data["len_total"] )
    val_si  = train_ei + 1 
    val_ei  = val_si + int( split_ratios[1] * Data["len_total"] )
    test_si   = val_ei + 1
    test_ei   = Data["len_total"] - 1
    

    #Guardo e imprimo por pantalla la cantidad de datos en cada conjunto
    Data["len_train"], Data["len_val"], Data["len_test"] = train_ei - train_si , val_ei - val_si , test_ei - test_si
    print("Number of instances in Train, Test and Val" , Data["len_train"], Data["len_val"], Data["len_test"] )    
    
    Data["xmin"], Data["xmax"] = Input.min()  , Input.max()
    Data["ymin"], Data["ymax"] = Target.min() , Target.max()
    
    
    Data['xmin']  = Input.min( axis = 0 )
    Data['xmax']  = Input.max( axis = 0 )
    Data['ymin']  = Target.min( axis = 0 )
    Data['ymax']  = Target.max( axis = 0 )
    
    Data['xmean'] = Input.mean( axis = 0 )
    Data['ymean'] = Target.mean( axis = 0 )
    Data['xstd']  = Input.std( axis = 0 )
    Data['ystd']  = Target.std( axis = 0 )
    
    TrainDataset = set_up_data(Data=Data, Input = Input[train_si:train_ei , :] , Target = Target[train_si:train_ei , :] )
    ValDataset   = set_up_data(Data=Data, Input = Input[val_si  :val_ei   , :] , Target = Target[val_si  :val_ei   , :] )
    TestDataset  = set_up_data(Data=Data, Input = Input[test_si :test_ei  , :] , Target = Target[test_si :test_ei  , :] )
    
    return Data, TrainDataset , ValDataset , TestDataset

class set_up_data(Dataset):
    "Para utilizarse con el DataLoader de PyTorch"
    def __init__(self,Data, Input, Target):
        self.x_data = Input

        self.y_data = Target
        #Parametros para la normalizacion
        self.xmin, self.xmax = Data["xmin"], Data["xmax"]
        self.ymin, self.ymax = Data["ymin"], Data["ymax"]

        #Normalizacion de los datos
        self.x_data = norm( self.x_data, self.xmin, self.xmax)
        self.y_data = norm( self.y_data, self.ymin, self.ymax)

    def __len__(self):
        "Denoto el numero total de muestras"
        return self.x_data.shape[0]
        
    def __getitem__(self,index):
        x = torch.tensor(self.x_data[index,:], dtype=torch.float)
        y = torch.tensor(self.y_data[index,:], dtype=torch.float)
        return x, y

#Funciones de normalizacion
def norm( data , datamin , datamax ):
    #print( data.shape , (2.0*(data-datamin)/(datamax-datamin)-1.0).shape )
    return (2.0*(data-datamin)/(datamax-datamin)-1.0) #Normalizacion [-1,1]
    #return ((data-datamin)/(datamax-datamin)) #Normalizacion [0,1]
    return data
    
def denorm( data , datamin , datamax) :
    return (0.5*(data+1.0)*(datamax-datamin)+datamin) #Normalizacion [-1,1]
    #return ((data.T)*(datamax-datamin)+datamin).T #Normalizacion [0,1]
    return data


def get_pca( data ) :
    #Assuming rows corresponds to instances, and columns to variables.
    #https://medium.com/@nahmed3536/a-python-implementation-of-pca-with-numpy-1bbd3b21de2e
    mean_data = data.mean( axis = 0 )
    std_data  = data.std( axis = 0 )
    datas = (data - mean_data ) / std_data
    cmatrix = np.cov( datas , ddof = 1, rowvar = False)
    eigenvalues, P = np.linalg.eig( cmatrix )
    
    return mean_data , std_data , P

def from_x2pca( data , mean , std , P ) :
    
    std_data = ( data - mean ) / std #Standarize data
    trans_data = np.dot( std_data , P )
    
    return trans_data 

def from_pca2x( trans_data , mean , std , P ) :
    
    std_data = np.dot( trans_data , P.T )
    data = ( std_data * std ) + mean
    
    return data



