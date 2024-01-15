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


conf.GeneralConf['NatureName']='NatureR5_Den1_Freq4_Hradar'
out_filename='./npz/Sesitivity_experiment_ptrip2.0_multinf_LETKF_' + conf.GeneralConf['NatureName'] + '.npz'
#Define the source of the observations
conf.GeneralConf['ObsFile']='./data/Nature/'+conf.GeneralConf['NatureName']+'.npz'
    
conf.DAConf['ExpLength'] = None                           #None use the full nature run experiment. Else use this length.
conf.DAConf['NEns'] = 20                                  #Number of ensemble members
conf.DAConf['Twin'] = True                                #When True, model configuration will be replaced by the model configuration in the nature run.
conf.DAConf['Freq'] = 4                                   #Assimilation frequency (in number of time steps)
conf.DAConf['TSFreq'] = 4                                 #Intra window ensemble output frequency (for 4D Data assimilation)
conf.DAConf['LocScalesLETKF']=np.array([3.0,-1.0])        #Localization scale is space and time (negative means no localization)
conf.DAConf['LocScalesLETPF']=np.array([3.0,-1.0])        #Localization scale is space and time (negative means no localization)
conf.DAConf['BridgeParam']=0.0                            #Bridging parameter for the hybrid 0-pure LETKF, 1.0-pure ETPF

conf.DAConf['AddaptiveTemp']=False                        #Enable addaptive tempering time step in pseudo time.
conf.DAConf['AlphaTempScale'] = 2.0                       #Scale factor to obtain the tempering factors on each tempering iteration.
conf.DAConf['GrossCheckFactor'] = 7.0                     #Optimized gross error check
conf.DAConf['LowDbzPerThresh']  = 1.1                     #Optimized Low ref thresh.

AlphaTempList=[]
MaxRipSteps = 4

if RunTheExperiment  :

    results=list()
    
    mult_inf_range = np.arange(1.1,1.8,0.05)
    
    total_analysis_rmse = np.zeros( (len(mult_inf_range),MaxRipSteps) )
    total_analysis_sprd = np.zeros( (len(mult_inf_range),MaxRipSteps) )
    total_forecast_rmse = np.zeros( (len(mult_inf_range),MaxRipSteps) )
    total_forecast_sprd = np.zeros( (len(mult_inf_range),MaxRipSteps) )
    
    for iinf , mult_inf in enumerate( mult_inf_range ) :
        for inrip in range( MaxRipSteps )  :
            
            conf.DAConf['InfCoefs']=np.array([mult_inf,0.0,0.0,0.0,0.0,0.0,0.0])
            conf.DAConf['NRip']= inrip + 1
            
            results.append( ahm.assimilation_hybrid_run( conf ) )
            AlphaTempList.append( ahm.get_temp_steps( conf.DAConf['NRip'] , conf.DAConf['AlphaTempScale'] ) )
            print('Multiplicative Inflation',mult_inf)
            print('Rip iteraations',conf.DAConf['NRip'])
            print('AlphaTemp',AlphaTempList[-1])
            print('Analisis RMSE: ',np.mean(results[-1]['XASRmse']))
            print('Forecast RMSE: ',np.mean(results[-1]['XFSRmse']))
            print('Analisis SPRD: ',np.mean(results[-1]['XASSprd']))
            print('Forecast SPRD: ',np.mean(results[-1]['XFSSprd']))
            
            total_analysis_rmse[iinf,inrip] = np.mean(results[-1]['XASRmse'])
            total_forecast_rmse[iinf,inrip] = np.mean(results[-1]['XFSRmse'])
            total_analysis_sprd[iinf,inrip] = np.mean(results[-1]['XASSprd'])
            total_forecast_sprd[iinf,inrip] = np.mean(results[-1]['XFSSprd'])
            
    f=open(out_filename,'wb')
    pickle.dump([results,AlphaTempList,mult_inf_range,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd],f)
    f.close()
    
