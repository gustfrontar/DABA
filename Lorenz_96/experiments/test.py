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
import assimilation_riphybrid_module as ahm

if len(sys.argv) > 1 and sys.argv[1] == 'compute' :
   RunTheExperiment = True
   PlotTheExperiment = False
else                        :
   RunTheExperiment = False
   PlotTheExperiment = True


conf.GeneralConf['NatureName']='NatureR8_Den05_Freq8_Hlinear'
#out_filename='./npz/Sesitivity_experiment_orip_multinf_LETKF_' + conf.GeneralConf['NatureName'] + '.npz'
#Define the source of the observations
conf.GeneralConf['ObsFile']='./data/Nature/'+conf.GeneralConf['NatureName']+'.npz'
    
conf.DAConf['ExpLength'] = 1000                           #None use the full nature run experiment. Else use this length.
conf.DAConf['NEns'] = 20                                  #Number of ensemble members
conf.DAConf['Twin'] = True                                #When True, model configuration will be replaced by the model configuration in the nature run.
conf.DAConf['Freq'] = 8                                   #Assimilation frequency (in number of time steps)
conf.DAConf['TSFreq'] = 8                                 #Intra window ensemble output frequency (for 4D Data assimilation)
conf.DAConf['LocScalesLETKF']=np.array([3.0,-1.0])        #Localization scale is space and time (negative means no localization)
conf.DAConf['LocScalesLETPF']=np.array([3.0,-1.0])        #Localization scale is space and time (negative means no localization)
conf.DAConf['BridgeParam']=0.0                            #Bridging parameter for the hybrid 0-pure LETKF, 1.0-pure ETPF

conf.DAConf['AddaptiveTemp']=False                        #Enable addaptive tempering time step in pseudo time.
conf.DAConf['EnableTempering']=False                      #Enable tempered iterations. If False, then traditional RIP method is applied without using tempering.


conf.DAConf['InfCoefs']=np.array([1.07,0.0,0.0,0.0,0.0])
conf.DAConf['NRip']=2
            
results = ahm.assimilation_hybrid_run( conf ) 
                 

print('Analisis RMSE: ',np.mean(results['XASRmse']))
print('Forecast RMSE: ',np.mean(results['XFSRmse']))
print('Analisis SPRD: ',np.mean(results['XASSprd']))
print('Forecast SPRD: ',np.mean(results['XFSSprd']))
  