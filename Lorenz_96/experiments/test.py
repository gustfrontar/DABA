#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:50:17 2020
@author: jruiz
"""
import pickle
import sys
sys.path.append('../model/')
sys.path.append('../data_assimilation/')

import numpy as np
import sensitivity_conf_default as conf
import assimilation_hybrid_module as ahm

RunTheExperiment = True
PlotTheExperiment = False

np.random.seed(10)


conf.GeneralConf['NatureName']='NatureR5_Den1_Freq8_Hradar'
#Define the source of the observations
conf.GeneralConf['ObsFile']='./data/Nature/'+conf.GeneralConf['NatureName']+'.npz'
    
conf.DAConf['ExpLength'] = 5000                           #None use the full nature run experiment. Else use this length.
conf.DAConf['NEns'] = 20                                  #Number of ensemble members
conf.DAConf['Twin'] = True                                #When True, model configuration will be replaced by the model configuration in the nature run.
conf.DAConf['Freq'] = 8                                   #Assimilation frequency (in number of time steps)
conf.DAConf['TSFreq'] = 8                                 #Intra window ensemble output frequency (for 4D Data assimilation)
conf.DAConf['LocScalesLETKF']=np.array([3.0,-1.0])        #Localization scale is space and time (negative means no localization)
conf.DAConf['LocScalesLETPF']=np.array([3.0,-1.0])        #Localization scale is space and time (negative means no localization)
conf.DAConf['BridgeParam']=0.0                            #Bridging parameter for the hybrid 0-pure LETKF, 1.0-pure ETPF

conf.DAConf['AddaptiveTemp']=False                        #Enable addaptive tempering time step in pseudo time.

AlphaTempList=[np.array([1])]
NAlphaTemp = len( AlphaTempList )

if RunTheExperiment  :

    results=list()
    
    mult_inf_list = [1.01]
    
    total_analysis_rmse = np.zeros( (len(mult_inf_list),NAlphaTemp) )
    total_analysis_sprd = np.zeros( (len(mult_inf_list),NAlphaTemp) )
    total_forecast_rmse = np.zeros( (len(mult_inf_list),NAlphaTemp) )
    total_forecast_sprd = np.zeros( (len(mult_inf_list),NAlphaTemp) )
    for iinf , mult_inf in enumerate( mult_inf_list ) :
        for intemp , AlphaTemp in enumerate( AlphaTempList )  :
            
            conf.DAConf['InfCoefs']=np.array([mult_inf,0.0,0.0,0.0,0.0])
            conf.DAConf['AlphaTemp'] = AlphaTemp
            conf.DAConf['NTemp']=len(AlphaTemp)
            
            results.append( ahm.assimilation_hybrid_run( conf ) )
                 
            print('Multiplicative Inflation',mult_inf)
            print('Tempering iteraations',conf.DAConf['NTemp'])
            print('AlphaTemp',AlphaTemp)
            print('Analisis RMSE: ',np.mean(results[-1]['XASRmse']))
            print('Forecast RMSE: ',np.mean(results[-1]['XFSRmse']))
            print('Analisis SPRD: ',np.mean(results[-1]['XASSprd']))
            print('Forecast SPRD: ',np.mean(results[-1]['XFSSprd']))
            
            total_analysis_rmse[iinf,intemp] = np.mean(results[-1]['XASRmse'])
            total_forecast_rmse[iinf,intemp] = np.mean(results[-1]['XFSRmse'])
            total_analysis_sprd[iinf,intemp] = np.mean(results[-1]['XASSprd'])
            total_forecast_sprd[iinf,intemp] = np.mean(results[-1]['XFSSprd'])
            

