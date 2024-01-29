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
import assimilation_letkf_module as alm

if len(sys.argv) > 1 and sys.argv[1] == 'compute' :
   RunTheExperiment = True
   PlotTheExperiment = False
else                        :
   RunTheExperiment = False
   PlotTheExperiment = True


conf.GeneralConf['NatureName']='NatureR03_Den1_Freq4_Hradar'
out_filename='./npz/Sesitivity_experiment_multinfyloc_LETKF_' + conf.GeneralConf['NatureName'] + '.npz'
#Define the source of the observations
conf.GeneralConf['ObsFile']='./data/Nature/'+conf.GeneralConf['NatureName']+'.npz'
    
conf.DAConf['ExpLength'] = None                           #None use the full nature run experiment. Else use this length.
conf.DAConf['NEns'] = 20                                  #Number of ensemble members
conf.DAConf['Twin'] = True                                #When True, model configuration will be replaced by the model configuration in the nature run.
conf.DAConf['Freq'] = 4                                   #Assimilation frequency (in number of time steps)
conf.DAConf['TSFreq'] = 4                                 #Intra window ensemble output frequency (for 4D Data assimilation)
#conf.DAConf['LocScalesLETKF']=np.array([3.0,-1.0])        #Localization scale is space and time (negative means no localization)
conf.DAConf['LocScalesLETPF']=np.array([3.0,-1.0])        #Localization scale is space and time (negative means no localization)
conf.DAConf['BridgeParam']=0.0                            #Bridging parameter for the hybrid 0-pure LETKF, 1.0-pure ETPF

conf.DAConf['AddaptiveTemp']=False                        #Enable addaptive tempering time step in pseudo time.
conf.DAConf['AlphaTempScale'] = 2.0                       #Scale factor to obtain the tempering factors on each tempering iteration.
conf.DAConf['GrossCheckFactor'] = 150.0                   #Optimized gross error check
conf.DAConf['LowDbzPerThresh']  = 0.9                     #Optimized Low ref thresh.
conf.DAConf['NTemp'] = 1                                  #Number of tempering iterations.


if RunTheExperiment  :

    results=list()
    
    mult_inf_range   = np.arange(1.0,1.6,0.05)
    loc_scale_range = np.arange(0.5,5.0,0.5) 
    AlphaTempList = []
    
    total_analysis_rmse = np.zeros( (len(mult_inf_range),len(loc_scale_range)) )
    total_analysis_sprd = np.zeros( (len(mult_inf_range),len(loc_scale_range)) )
    total_forecast_rmse = np.zeros( (len(mult_inf_range),len(loc_scale_range)) )
    total_forecast_sprd = np.zeros( (len(mult_inf_range),len(loc_scale_range)) )
    
    for iinf , mult_inf in enumerate( mult_inf_range ) :
        for iloc , loc_scale in enumerate( loc_scale_range ) :
            
            conf.DAConf['InfCoefs']=np.array([mult_inf,0.0,0.0,0.0,0.0])
            conf.DAConf['LocScalesLETKF'] = np.array([loc_scale,-1.0])
            
            results.append( alm.assimilation_letkf_run( conf ) )
            AlphaTempList.append( alm.get_temp_steps( conf.DAConf['NTemp'] , conf.DAConf['AlphaTempScale'] ) )
                 
            print('Multiplicative Inflation',mult_inf)
            print('Localization Scale',loc_scale)
            print('AlphaTemp',AlphaTempList[-1])
            print('Analisis RMSE: ',np.mean(results[-1]['XASRmse']))
            print('Forecast RMSE: ',np.mean(results[-1]['XFSRmse']))
            print('Analisis SPRD: ',np.mean(results[-1]['XASSprd']))
            print('Forecast SPRD: ',np.mean(results[-1]['XFSSprd']))
            
            total_analysis_rmse[iinf,iloc] = np.mean(results[-1]['XASRmse'])
            total_forecast_rmse[iinf,iloc] = np.mean(results[-1]['XFSRmse'])
            total_analysis_sprd[iinf,iloc] = np.mean(results[-1]['XASSprd'])
            total_forecast_sprd[iinf,iloc] = np.mean(results[-1]['XFSSprd'])
            
    f=open(out_filename,'wb')
    pickle.dump([results,mult_inf_range,loc_scale_range,AlphaTempList,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd],f)
    f.close()
    
if PlotTheExperiment  :

    f=open(out_filename,'rb')
    [results,inf_range,AlphaTempList,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
    f.close()

    import matplotlib.pyplot as plt

    plt.pcolormesh(np.arange(NAlphaTemp),inf_range,total_analysis_rmse)
    plt.colorbar()
    plt.title('Analysis Rmse')
    plt.xlabel('Tempering Iterantions')
    plt.ylabel('Multiplicative Inflation')
    plt.show()

    plt.figure()
    plt.plot(inf_range,total_analysis_rmse[:,0]);plt.plot(inf_range,total_analysis_rmse[:,1]);plt.plot(inf_range,total_analysis_rmse[:,2]);plt.plot(inf_range,total_analysis_rmse[:,3])
    plt.xlabel('Inflation')
    plt.ylabel('Analysis RMSE')
    plt.show()

    plt.figure()
    plt.plot(total_analysis_sprd[:,0],total_analysis_rmse[:,0]);plt.plot(total_analysis_sprd[:,1],total_analysis_rmse[:,1]);plt.plot(total_analysis_sprd[:,-1],total_analysis_rmse[:,-1])
    plt.show()



