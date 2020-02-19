#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:50:17 2020
Simple script to test configurations fast.
@author: jruiz
"""

import sys
sys.path.append('../model/')
sys.path.append('../data_assimilation/')

import numpy as np
import assimilation_riphybrid_module as hymrip
import assimilation_hybrid_module as hym
import assimilation_conf_GM_default as conf         #Load the experiment configuration

conf.GeneralConf['NatureName']='NatureR4_Den05_Freq8_Hlogaritmic'
conf.GeneralConf['ObsFile']='./data/Nature/'+conf.GeneralConf['NatureName']+'.npz'
    
conf.DAConf['ExpLength'] = 1000                           #None use the full nature run experiment. Else use this length.
conf.DAConf['NEns'] = 20                                  #Number of ensemble members
conf.DAConf['LocScalesLETKF']=np.array([3.0,-1.0])        #Localization scale is space and time (negative means no localization)
conf.DAConf['LocScalesLETPF']=np.array([3.0,-1.0])        #Localization scale is space and time (negative means no localization)

conf.DAConf['BetaCoef']=0.4                                 #Scaling parameter for the Gaussian Kernel in the Gaussian mixture prior.
conf.DAConf['GammaCoef']=0.2                                #Nudging parameter to uniform weigths in order to avoid weigth degeneracy. 
conf.DAConf['ResamplingType']=2                             #Resampling: 1-Liu 2016, 2-Acevedo et al. 2016, 3-NETPF without rotation, 4-NETPF with rotation.

#Introduce a model error in the model used for the assimilation experiment.
conf.DAConf['Twin'] = True                                  #When True, model configuration will be replaced by the model configuration in the nature run.
conf.ModelConf['Coef']=np.array([8.0])                      #Coefficient of parametrized forcing (polynom coefficients starting from coef[0]*x^0 + coef[1]*x ... )
conf.DAConf['AddaptiveTemp']=True                           #Enable addaptive tempering time step in pseudo time.
conf.DAConf['NTemp']=1                                      #Number of temper iterations 
conf.DAConf['InfCoefs']=np.array([1.2,0.0,0.0,0.0,0.0])
conf.DAConf['RejuvParam']=0.0                               #Global particle rejuvenestion (For the ETPF only)
conf.DAConf['BridgeParam']=0.0 
conf.DAConf['NRip']=2
conf.DAConf['EnableTempering']=True

results =  hymrip.assimilation_hybrid_run( conf ) 
                 
print('Multiplicative Inflation',conf.DAConf['InfCoefs'][0])
print('Tempering iteraations',conf.DAConf['NTemp'])
print('Analisis RMSE: ',np.mean(results['XASRmse']))
print('Forecast RMSE: ',np.mean(results['XFSRmse']))
print('Analisis SPRD: ',np.mean(results['XASSprd']))
print('Forecast SPRD: ',np.mean(results['XFSSprd']))
 

conf.DAConf['InfCoefs']=np.array([1.2,0.0,0.0,0.0,0.0])           
conf.DAConf['BridgeParam']=0.0                        
results =  hym.assimilation_hybrid_run( conf ) 
                
print('Multiplicative Inflation',conf.DAConf['InfCoefs'][0])
print('Tempering iteraations',conf.DAConf['NTemp'])
print('Analisis RMSE: ',np.mean(results['XASRmse']))
print('Forecast RMSE: ',np.mean(results['XFSRmse']))
print('Analisis SPRD: ',np.mean(results['XASSprd']))
print('Forecast SPRD: ',np.mean(results['XFSSprd']))