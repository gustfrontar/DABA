#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:06:26 2024

@author: jruiz
"""
import numpy as np
import torch.nn as nn
from scipy.stats import spearmanr
import torch
import set_dataset as ds

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# class CELoss(nn.Module):
#     def __init__(self):
#         super(CELoss, self).__init__()

#     def forward(self, predictions, targets):
#         #From Qi and Majda Using machine learning to predic extreme events in complex systems
#         ptilde_pos = nn.functional.softmax( predictions )
#         ttilde_pos = nn.functional.softmax( targets )
#         ptilde_neg = nn.functional.softmax( -predictions )
#         ttilde_neg = nn.functional.softmax( -targets )
        
#         kl_pos = torch.sum( ttilde_pos * ( torch.log( ttilde_pos ) - torch.log( ptilde_pos ) ) )
#         kl_neg = torch.sum( ttilde_neg * ( torch.log( ttilde_neg ) - torch.log( ptilde_neg ) ) )
            
#         return kl_pos + kl_neg

def CELoss( predictions , targets ) :
    
   #From Qi and Majda Using machine learning to predic extreme events in complex systems
   ptilde_pos = nn.functional.softmax( predictions )
   ttilde_pos = nn.functional.softmax( targets )
   ptilde_neg = nn.functional.softmax( -predictions )
   ttilde_neg = nn.functional.softmax( -targets )
        
   kl_pos = torch.sum( ttilde_pos * ( torch.log( ttilde_pos ) - torch.log( ptilde_pos ) ) )
   kl_neg = torch.sum( ttilde_neg * ( torch.log( ttilde_neg ) - torch.log( ptilde_neg ) ) )
            
   return kl_pos + kl_neg

def WMSELoss( predictions , targets ) :
   alpha = 2.5
   #Use softmax functional to give more weight to extreme values within the target. 
   loss = torch.sum( torch.pow( predictions - targets , 2 ) * nn.functional.softmax( alpha * torch.abs( targets ) ) ) 
            
   return loss

def SMSELoss( predictions , targets ) :
    
   #Use softmax functional to give more weight to extreme values within the target.
   nvar = predictions.shape[1]
   MSE = torch.sum( torch.pow( predictions - targets , 2 ) ) 
   Star = torch.abs( torch.fft.rfft( targets ) )                
   Spre = torch.abs( torch.fft.rfft( predictions ) )
   SMSE = torch.sum( torch.pow(Star - Spre ,2  ) ) / nvar
         
   return SMSE * 8.0 + MSE  

def SPECLoss( predictions , targets ) :
    
   #Use softmax functional to give more weight to extreme values within the target.
   Star = torch.fft.rfft( targets )                
   Spre = torch.fft.rfft( predictions )
   AMPMSE = torch.sum( torch.pow( torch.abs( Star ) - torch.abs( Spre ) ,2  ) )  #Amplitude loss
   ANGMSE = torch.sum( torch.pow( torch.sin( 0.5* (torch.angle( Star ) -  torch.angle( Spre ) ) ) ,2  ) ) #Phase loss 
   #MSE = torch.sum( torch.pow( predictions - targets , 2 ) )
         
   return ANGMSE + AMPMSE #+ MSE 


def get_model( dim , expand_factor ) :

    nnmodel = nn.Sequential(
        nn.Linear(in_features = int(dim), out_features = int(dim * expand_factor ), bias=True),
        nn.ReLU(),
        nn.Linear(in_features = int(dim * expand_factor ), out_features = int(dim * expand_factor ), bias=True),
        nn.ReLU(),
        nn.Linear(in_features = int(dim * expand_factor ), out_features = int(dim * expand_factor ), bias=True),
        nn.ReLU(),
        nn.Linear(in_features = int(dim * expand_factor ), out_features = int(dim * expand_factor ), bias=True),
        nn.ReLU(),
        nn.Linear(in_features = int(dim * expand_factor ), out_features = int(dim), bias=True)
    
     )
    
    nnmodel.to(device) #Cargamos en memoria, de haber disponible GPU se computa por ahí.
     
    return nnmodel 

def initialize_weights( nnmodel ):
    for m in nnmodel.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            #torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.constant_(m.bias, 0)

    return nnmodel

def train_model( model , learning_rate , TrainDataloader , ValDataloader , max_epochs , loss = 'MSE') :
    
    #Listas donde guardamos loss de entrenamiento, y para el de validación la loss y las métricas de evaluación.
    RMSE_train , RMSE_val = [] , []
    BIAS_train , BIAS_val = [] , []
    CorP_train , CorP_val = [] , []
    CorS_train , CorS_val = [] , [] 
    Loss_train , Loss_val = [] , [] 
    
    optimizer = torch.optim.Adam( model.parameters() , lr=learning_rate  )
    
    if loss == 'CE_Loss' :
      optim_loss = CELoss
    elif loss == 'WMSE'  :
      optim_loss = WMSELoss 
    elif loss == 'SMSE' :
      optim_loss = SMSELoss
    elif loss == 'SPEC' :
      optim_loss = SPECLoss 
    elif loss == 'MSE'  :
      optim_loss = torch.nn.MSELoss(reduction='mean')  
      
    for epoch in range( max_epochs ):
        print('Epoca: '+ str(epoch+1) + ' de ' + str(max_epochs) )
        
        #Entrenamiento del modelo        
        model.train()  #Esto le dice al modelo que se comporte en modo entrenamiento.
    
        sum_loss = 0.0
        batch_counter = 0
    
        # Iteramos sobre los minibatches. 
        for TrainInput , TrainTarget in TrainDataloader :
            #Enviamos los datos a la memoria.
            TrainInput , TrainTarget = TrainInput.to(device), TrainTarget.to(device)
            #-print( 'Batch ' + str(batch_counter) )
    
            optimizer.zero_grad()
    
            TrainOutput = model( TrainInput )
            
            Loss = optim_loss( TrainOutput.float(), TrainTarget.float())
                        
            Loss.backward()
            optimizer.step()
                        
            batch_counter += 1
            sum_loss = sum_loss + Loss.item()
    
        #Calculamos la loss media sobre todos los minibatches 
        Loss_train.append( sum_loss / batch_counter )
    
        #Calculamos la funcion de costo para la muestra de validacion.
        ValInput , ValTarget = next( iter( ValDataloader ) )
        ValInput , ValTarget = ValInput.detach().to(device) , ValTarget.detach().to(device) 
        with torch.no_grad():
          ValOutput = model( ValInput ).detach()
    
        Loss_val.append( optim_loss( ValOutput , ValTarget ).item() )
        
        #Calculo de la loss de la epoca
        print('Loss Train: ', str( Loss_train[epoch] ) )
        print('Loss Val:  ', str( Loss_val[epoch] ) )
    
        ###################################
        np_target_val = ds.denorm( ValTarget.numpy() , ValDataloader.dataset.ymin, ValDataloader.dataset.ymax )
        np_output_val = ds.denorm( ValOutput.numpy() , ValDataloader.dataset.ymin, ValDataloader.dataset.ymax )
        
        np_target_train = ds.denorm( TrainTarget.detach().numpy() , TrainDataloader.dataset.ymin, TrainDataloader.dataset.ymax )
        np_output_train = ds.denorm( TrainOutput.detach().numpy() , TrainDataloader.dataset.ymin, TrainDataloader.dataset.ymax )
      
        #Calculo de métricas RMSE, BIAS, Correlacién de Pearson y Spearman
        RMSE_train.append( rmse( np_output_train , np_target_train ) )
        BIAS_train.append( bias( np_output_train , np_target_train ) )
        CorP_train.append( corr_P( np_output_train , np_target_train ) )
        CorS_train.append( corr_S( np_output_train , np_target_train ) ) 

        #Calculo de métricas RMSE, BIAS, Correlacién de Pearson y Spearman
        RMSE_val.append( rmse( np_output_val , np_target_val ) )
        BIAS_val.append( bias( np_output_val , np_target_val ) )
        CorP_val.append( corr_P( np_output_val , np_target_val ) )
        CorS_val.append( corr_S( np_output_val , np_target_val ) ) 
        
    Verification = dict()
    Verification['RMSE_val'] = np.array( RMSE_val )   
    Verification['BIAS_val'] = np.array( BIAS_val )
    Verification['CorP_val'] = np.array( CorP_val )
    Verification['CorS_val'] = np.array( CorS_val )    

    Verification['RMSE_train'] = np.array( RMSE_train )   
    Verification['BIAS_train'] = np.array( BIAS_train )
    Verification['CorP_train'] = np.array( CorP_train )
    Verification['CorS_train'] = np.array( CorS_train )
        
    return model , Verification    

def model_eval( MyModel , MyDataloader  ) :
    
    Input , Target = next( iter( MyDataloader ) )
    Input , Target = Input.detach().to(device) , Target.detach().to(device) 
    with torch.no_grad():
         Output = MyModel( Input ) .detach().numpy()  
    Input , Target = Input.numpy() , Target.numpy()     
    #Denormalize data.     
    Input  = ds.denorm( Input , MyDataloader.dataset.ymin, MyDataloader.dataset.ymax )
    Target = ds.denorm( Target , MyDataloader.dataset.ymin, MyDataloader.dataset.ymax )
    Output = ds.denorm( Output , MyDataloader.dataset.ymin, MyDataloader.dataset.ymax )
    
    
    return Input , Output , Target 


          
def rmse( modeldata , targetdata ) :
    return np.sqrt( np.mean( (modeldata.flatten() - targetdata.flatten()) ** 2 ) )

def bias( modeldata , targetdata ) :
    return np.mean( modeldata.flatten() - targetdata.flatten() )

def corr_P( modeldata , targetdata ) :    
    return np.corrcoef( modeldata.flatten() , targetdata.flatten() )[0,1]

def corr_S( modeldata , targetdata ) :    
    return spearmanr( modeldata.flatten() , targetdata.flatten() )[0]        
        