#%%
#!/usr/bin/env python
# coding: utf-8

# Inicializacion. Cargamos los modulos necesarios

import sys

sys.path.append("../")

#Importamos todos los modulos que van a ser usados en esta notebook
from tqdm import tqdm
import numpy as np
import Lorenz_63 as model
import Lorenz_63_DA as da


def da_cycle_tempered_hybrid( da_exp ) :
    #This function performs a full data assimilation run for the tempered hybrid method.
    
    
    #This ensures that two experiments with the same random_seed will produce the same 
    #sequence of observations and random number for particle rejuvenation.
    
    np.random.seed( da_exp['random_seed'] )
    

    #Get forward operators from the input dictionary. 
    forward_operator = da_exp['forward_operator']
    forward_operator_tl = da_exp['forward_operator_tl']
    
    #%%
    #------------------------------------------------------------
    # Generamos la simulacion "verdad"
    #------------------------------------------------------------
    
    # Generamos la verdadera evolucion del sistema ("nature run")
    # Integramos el modelo durante varios pasos de tiempo para que la solucion converja al atractor.
    
    x=np.copy(da_exp['x0'])
    for i in range(da_exp['numtrans'])  :
       x = model.forward_model( x , da_exp['p'] , da_exp['dt'] )
        
    # Integramos la simulacion verdad
    # El resultado es almacenado en un array de numpy "state" con dimension (numstep,3)
    
    da_exp['state']=np.zeros((da_exp['numstep'],da_exp['nvars']))
    
    for i  in range( da_exp['numstep'] ) :
        for j in range( da_exp['bst'] )      :
            x = model.forward_model( x , da_exp['p'] , da_exp['dt'] )
          
        da_exp['state'][i,:]=x
    
    #%%
    #------------------------------------------------------------
    # Simulamos las observaciones y generamos las matrices que necesitamos
    # para la asimilacion
    #------------------------------------------------------------
    
    # Simulamos las observaciones 
    
    # initialize the matrix operator for data assimilation
    da_exp = da.gen_obs( forward_operator , da_exp )
    
    #%%    
    #------------------------------------------------------------
    # Corremos el ciclo de asimilacion
    #------------------------------------------------------------
    
    #Simulamos N ciclos de asimilacion en los cuales se asimilan las observaciones simuladas
    
    #Iniciamos el ciclo de asimilacion
    
    #Creamos el array que contiene a los analisis
    da_exp['statea'] = np.zeros((da_exp['numstep'],da_exp['nvars']))
    #Creamos el array que contiene a los pronosticos
    da_exp['statef'] = np.zeros((da_exp['numstep'],da_exp['nvars'],da_exp['forecast_length'])) + np.nan
    
    #Creamos el array que contiene al ensamble de analisis
    da_exp['stateaens'] = np.zeros((da_exp['numstep'],da_exp['nvars'],da_exp['EnsSize']))
    #Creamos el array que contiene al ensamble de pronosticos
    da_exp['statefens'] = np.zeros((da_exp['numstep'],da_exp['nvars'],da_exp['EnsSize'],da_exp['forecast_length'])) + np.nan
    
    #Creamos el array que contiene la diserpsion del analisis y del forecast
    da_exp['stateasprd'] = np.zeros((da_exp['numstep'],da_exp['nvars']))
    #Creamos el array que contiene al ensamble de pronosticos
    da_exp['statefsprd'] = np.zeros((da_exp['numstep'],da_exp['nvars'],da_exp['forecast_length'])) + np.nan
    
    da_exp['w'] = np.zeros((da_exp['numstep'],da_exp['EnsSize'])) 

    
    #Inicializamos el ciclo desde la media "climatologica" del sistema. Es decir no tenemos informacion precisa
    #de donde esta el sistema al tiempo inicial.
    
    for iens in range( da_exp['EnsSize']) :
        da_exp['stateaens'][0,:,iens] = np.nanmean( da_exp['state'] , 0 ) + np.random.multivariate_normal(np.zeros(da_exp['nvars']),da_exp['P0'])
        
    #Inicializamos el ensamble tomando una muestra random con media 0 y varianza P.
    [ mean , pert ] = da.mean_and_perts( da_exp['stateaens'][0,:,:] ) #Get the ensemble mean and perturbations from the full ensemble.
    
    da_exp['statea'][0,:] = mean
    da_exp['w'][0,:] = np.ones( da_exp['EnsSize'] ) * ( 1.0 / da_exp['EnsSize'] )
    
    
    for i in tqdm( range(1,da_exp['numstep']) ) :
    #for i in range(1,da_exp['numstep']) :
        #print(i)
        
        #Vamos a hacer el pronostico de x con el modelo no lineal y el
        #pronostico de la covarianza con el tangente lineal y el adjunto.
        #Integrate the forward non-linear model to obtain the first-guess
           
        for iens in range( da_exp['EnsSize'] ) :
            x=np.copy( da_exp['stateaens'][i-1,:,iens] )
            #Integramos el i-esimo miembro del ensamble.
            for k in range(da_exp['forecast_length']) :
                #Aplico la inflacion aditiva antes de integrar las ecuaciones.
                x = x + np.random.multivariate_normal(np.zeros(da_exp['nvars']),da_exp['Q'])
                for j in range(da_exp['bst'])  :
                    x = model.forward_model( x ,
                    da_exp['p'] , da_exp['dt'] )
                
              
                if ( i + k < da_exp['numstep'] ) :  #Solo para cerciorarme de que el pronostico no queda fuera del rango del experimento.
                   da_exp['statefens'][i+k,:,iens,k]=x
        
        #Construct and store the evolution of the error covariance matrix and the forecast ensemble mean
        for k in range(da_exp['forecast_length']) :
            if ( i + k < da_exp['numstep'] ) : 
               [ mean , pert ] = da.mean_and_perts( da_exp['statefens'][i+k,:,:,k] )
               da_exp['statef'][i+k,:,k]=mean
    
        #Reemplazamos la llamada a la funcion de asimilacion por un ciclo que se repite tantas veces
        #como ciclos de temperado hayamos definido. 
        gamma = 1.0/da_exp['ntemp']
        Rtemp=da_exp['R'] / gamma
        rejuv_param_temp = da_exp['rejuv_param'] * gamma
        stateens = np.copy( da_exp['statefens'][i,:,:,0] )
        rejuv_rand = np.random.randn( da_exp['EnsSize'] , da_exp['EnsSize'] ) / np.sqrt( da_exp['EnsSize'] - 1 )
        for itemp in range( da_exp['ntemp'])  :  
    
            if da_exp['bridge'] < 1 :  #If not then ETKF will not be performed
                multinf_temp = np.power ( da_exp['multinf'] , gamma * ( 1 - da_exp['bridge'] ) ) 
                Rtemp_ETKF = Rtemp / ( 1.0 - da_exp['bridge'] ) 
                [ stateens , state , _ , _ , _ ] =da.analysis_update_ETKF(da_exp['yobs'][i,:], stateens , forward_operator , Rtemp_ETKF , multinf_temp )
            if da_exp['bridge'] > 0 :  #If not then ETPF will not be performed
                Rtemp_ETPF = Rtemp / ( da_exp['bridge'] )
                rejuv_param_temp = da_exp['rejuv_param'] * np.sqrt( gamma ) * da_exp['bridge'] 
                [ stateens , state , _ , _ , _ , da_exp['w'][i,:] , S ] =da.analysis_update_ETPF_2ndord(da_exp['yobs'][i,:], stateens ,forward_operator, Rtemp_ETPF , rejuv_param_temp , rejuv_perts = da_exp['statefens'][i,:,:,0] )
    
        da_exp['stateaens'][i,:,:] = np.copy( stateens )
        da_exp['statea'][i,:] = np.copy( state )
    
    da_exp['stateastd'] = np.std( da_exp['stateaens'] , 2 )
    da_exp['statefstd'] = np.std( da_exp['statefens'] , 2 )
    
    #%%
    #------------------------------------------------------------
    # Verificacion del ciclo de los resultados
    #------------------------------------------------------------
    
    #Calculamos los errores excluyendo los primeros spin_up ciclos de asimilacion.
        
    da_exp = da.analysis_verification( forward_operator , da_exp ) #Calculamos el RMSE y BIAS del analisis.
    
    #Removemos las key mas pesadas del diccionario antes de devolver el diccionario a la funcion que colecta todo.
    del da_exp['statefens']
    del da_exp['stateaens']
    
    return da_exp


    





