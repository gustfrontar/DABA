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

NatureList = [ 'NatureR4_Den05_Freq8_Hlogaritmic'   ,
               'NatureR8_Den05_Freq8_Hlinear'       ,
               'NatureR4_Den05_Freq8_Hcuadratic'    ,
               'NatureR1_Den1_Freq4_Hlinear'        ,
               'NatureR1_Den05_Freq25_Hlinear'
               ]
   
#Define the source of the observations
outfile_prefix = './npz/Sesitivity_experiment_bridge_multinf_hybrid_'

conf.DAConf['ExpLength'] = None                            #None use the full nature run experiment. Else use this length.
conf.DAConf['NEns'] = 20                                   #Number of ensemble members
conf.DAConf['Twin'] = True                                 #When True, model configuration will be replaced by the model configuration in the nature run.
conf.DAConf['LocScalesLETKF']=np.array([3.0,-1.0])         #Localization scale is space and time (negative means no localization)
conf.DAConf['LocScalesLETPF']=np.array([3.0,-1.0])         #Localization scale is space and time (negative means no localization)

conf.DAConf['BetaCoef']=0.6                                 #Scaling parameter for the Gaussian Kernel in the Gaussian mixture prior.
conf.DAConf['GammaCoef']=0.2                                #Nudging parameter to uniform weigths in order to avoid weigth degeneracy. 
conf.DAConf['ResamplingType']=2                             #Resampling: 1-Liu 2016, 2-Acevedo et al. 2016, 3-NETPF without rotation, 4-NETPF with rotation.

conf.DAConf['AddaptiveTemp']=True                          #Enable addaptive tempering time step in pseudo time.
conf.DAConf['EnableTempering']=True                         #Enable tempered iterations. If False, then traditional RIP method is applied without using tempering.

conf.DAConf['NTemp']=3


for Nature in NatureList :

    conf.GeneralConf['NatureName'] = Nature
    conf.GeneralConf['ObsFile']='./data/Nature/'+conf.GeneralConf['NatureName']+'.npz'
    
    out_filename= outfile_prefix + conf.GeneralConf['NatureName'] + '.npz'

    results=list()

    if Nature == 'NatureR4_Den05_Freq8_Hlogaritmic' :
       mult_inf_range = np.arange(1.2,1.45,0.05)
    else                                            :
       mult_inf_range = np.arange(1.01,1.16,0.03)


    bridge_range = np.arange(0.0,1.1,0.1)

    total_analysis_rmse = np.zeros( (len(mult_inf_range),len(bridge_range)) )
    total_analysis_sprd = np.zeros( (len(mult_inf_range),len(bridge_range)) )
    total_forecast_rmse = np.zeros( (len(mult_inf_range),len(bridge_range)) )
    total_forecast_sprd = np.zeros( (len(mult_inf_range),len(bridge_range)) )

    for iinf , mult_inf in enumerate( mult_inf_range ) :
        for ibridge , bridge in enumerate( bridge_range )  :
        
            conf.DAConf['InfCoefs']=np.array([mult_inf,0.0,0.0,0.0,0.0])
            conf.DAConf['BridgeParam']=bridge
            print(conf.DAConf['BridgeParam'])
        
            results.append( ahm.assimilation_hybrid_run( conf ) )
             
            print('Multiplicative Inflation',mult_inf)
            print('Beta',bridge)
            print('Analisis RMSE: ',np.mean(results[-1]['XASRmse']))
            print('Forecast RMSE: ',np.mean(results[-1]['XFSRmse']))
            print('Analisis SPRD: ',np.mean(results[-1]['XASSprd']))
            print('Forecast SPRD: ',np.mean(results[-1]['XFSSprd']))
        
            total_analysis_rmse[iinf,ibridge] = np.mean(results[-1]['XASRmse'])
            total_forecast_rmse[iinf,ibridge] = np.mean(results[-1]['XFSRmse'])
            total_analysis_sprd[iinf,ibridge] = np.mean(results[-1]['XASSprd'])
            total_forecast_sprd[iinf,ibridge] = np.mean(results[-1]['XFSSprd'])
        
    f=open(out_filename,'wb')
    pickle.dump([results,mult_inf_range,bridge_range,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd],f)
    f.close()



