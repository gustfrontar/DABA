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

conf.GeneralConf['NatureName']='NatureR4_Den05_Freq8_Hlogaritmic'
out_filename='Sesitivity_experiment_bridging_addinf_Ens100_' + conf.GeneralConf['NatureName'] + '.npz'
    
conf.DAConf['ExpLength'] = None                           #None use the full nature run experiment. Else use this length.
conf.DAConf['NEns'] = 100                                  #Number of ensemble members
conf.DAConf['Twin'] = True                                #When True, model configuration will be replaced by the model configuration in the nature run.
conf.DAConf['Freq'] = 8                                   #Assimilation frequency (in number of time steps)
conf.DAConf['TSFreq'] = 8                                 #Intra window ensemble output frequency (for 4D Data assimilation)
conf.DAConf['LocScalesLETKF']=np.array([2.0,-1.0])        #Localization scale is space and time (negative means no localization)
conf.DAConf['LocScalesLETPF']=np.array([2.0,-1.0])        #Localization scale is space and time (negative means no localization)
conf.DAConf['BridgeParam']=0.0                            #Bridging parameter for the hybrid 0-pure LETKF, 1.0-pure ETPF
conf.DAConf['NTemp']=1                   

if RunTheExperiment  :

    results=list()
    
    add_inf_range = np.arange(0.01,0.1,0.01)
    
    bridge_param_range = np.arange(0,1.1,0.1)
    
    total_analysis_rmse = np.zeros( (len(add_inf_range),len(bridge_param_range)) )
    total_analysis_sprd = np.zeros( (len(add_inf_range),len(bridge_param_range)) )
    total_forecast_rmse = np.zeros( (len(add_inf_range),len(bridge_param_range)) )
    total_forecast_sprd = np.zeros( (len(add_inf_range),len(bridge_param_range)) )
    
    for iinf , add_inf in enumerate( add_inf_range ) :
        for ibridge , bridge_param in enumerate( bridge_param_range )  :            
            conf.DAConf['InfCoefs']=np.array([1.0,0.0,0.0,0.0,add_inf])
            conf.DAConf['BridgeParam']=bridge_param             
            results.append( ahm.assimilation_hybrid_run( conf ) )                 
            print('Additive Inflation',add_inf)
            print('Bridging Parameter',bridge_param)
            print('Analisis RMSE: ',np.mean(results[-1]['XASRmse']))
            print('Forecast RMSE: ',np.mean(results[-1]['XFSRmse']))
            print('Analisis SPRD: ',np.mean(results[-1]['XASSprd']))
            print('Forecast SPRD: ',np.mean(results[-1]['XFSSprd']))
            
            total_analysis_rmse[iinf,ibridge] = np.mean(results[-1]['XASRmse'])
            total_forecast_rmse[iinf,ibridge] = np.mean(results[-1]['XFSRmse'])
            total_analysis_sprd[iinf,ibridge] = np.mean(results[-1]['XASSprd'])
            total_forecast_sprd[iinf,ibridge] = np.mean(results[-1]['XFSSprd'])
            
    f=open(out_filename,'wb')
    pickle.dump([results,add_inf_range,bridge_param_range,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd],f)
    f.close()
    
if PlotTheExperiment  :
    
    f=open(out_filename,'rb')
    [results,add_inf_range,bridge_param_range,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
    f.close()
    
    import matplotlib.pyplot as plt 

    plt.pcolormesh(bridge_param_range,add_inf_range,total_analysis_rmse)
    plt.colorbar()
    plt.title('Analysis Rmse')
    plt.xlabel('Bridge Parameter')
    plt.ylabel('Additive Inflation')
    plt.show()


