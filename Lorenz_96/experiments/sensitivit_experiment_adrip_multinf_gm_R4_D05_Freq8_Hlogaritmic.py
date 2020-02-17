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
import assimilation_ripgm_module as ahm

if len(sys.argv) > 1 and sys.argv[1] == 'compute' :
   RunTheExperiment = True
   PlotTheExperiment = False
else                        :
   RunTheExperiment = False
   PlotTheExperiment = True


conf.GeneralConf['NatureName']='NatureR4_Den05_Freq8_Hlogaritmic'
out_filename='./npz/Sesitivity_experiment_adrip_multinf_gm_' + conf.GeneralConf['NatureName'] + '.npz'
#Define the source of the observations
conf.GeneralConf['ObsFile']='./data/Nature/'+conf.GeneralConf['NatureName']+'.npz'
    
conf.DAConf['ExpLength'] = None                           #None use the full nature run experiment. Else use this length.
conf.DAConf['NEns'] = 20                                  #Number of ensemble members
conf.DAConf['Twin'] = True                                #When True, model configuration will be replaced by the model configuration in the nature run.
conf.DAConf['Freq'] = 8                                   #Assimilation frequency (in number of time steps)
conf.DAConf['TSFreq'] = 8                                 #Intra window ensemble output frequency (for 4D Data assimilation)
conf.DAConf['LocScalesLETKF']=np.array([3.0,-1.0])        #Localization scale is space and time (negative means no localization)
conf.DAConf['LocScalesLETPF']=np.array([3.0,-1.0])        #Localization scale is space and time (negative means no localization)

conf.DAConf['BetaCoef']=0.6                                 #Scaling parameter for the Gaussian Kernel in the Gaussian mixture prior.
conf.DAConf['GammaCoef']=0.2                                #Nudging parameter to uniform weigths in order to avoid weigth degeneracy. 
conf.DAConf['ResamplingType']=2                             #Resampling: 1-Liu 2016, 2-Acevedo et al. 2016, 3-NETPF without rotation, 4-NETPF with rotation.

conf.DAConf['AddaptiveTemp']=True                        #Enable addaptive tempering time step in pseudo time.
conf.DAConf['EnableTempering']=True                      #Enable tempered iterations. If False, then traditional RIP method is applied without using tempering.

if RunTheExperiment  :

    
    results=list()
    
    mult_inf_range = np.arange(1.15,1.35,0.02)
    
    nrip_range = np.arange(1,5,1)
    
    total_analysis_rmse = np.zeros( (len(mult_inf_range),len(nrip_range)) )
    total_analysis_sprd = np.zeros( (len(mult_inf_range),len(nrip_range)) )
    total_forecast_rmse = np.zeros( (len(mult_inf_range),len(nrip_range)) )
    total_forecast_sprd = np.zeros( (len(mult_inf_range),len(nrip_range)) )
    
    for iinf , mult_inf in enumerate( mult_inf_range ) :
        for inrip , nrip in enumerate( nrip_range )  :
            
            conf.DAConf['InfCoefs']=np.array([mult_inf,0.0,0.0,0.0,0.0])
            conf.DAConf['NRip']=int(nrip) 
            
            results.append( ahm.assimilation_gm_run( conf ) )
                 
            print('Multiplicative Inflation',mult_inf)
            print('Rip iteraations',nrip)
            print('Analisis RMSE: ',np.mean(results[-1]['XASRmse']))
            print('Forecast RMSE: ',np.mean(results[-1]['XFSRmse']))
            print('Analisis SPRD: ',np.mean(results[-1]['XASSprd']))
            print('Forecast SPRD: ',np.mean(results[-1]['XFSSprd']))
            
            total_analysis_rmse[iinf,inrip] = np.mean(results[-1]['XASRmse'])
            total_forecast_rmse[iinf,inrip] = np.mean(results[-1]['XFSRmse'])
            total_analysis_sprd[iinf,inrip] = np.mean(results[-1]['XASSprd'])
            total_forecast_sprd[iinf,inrip] = np.mean(results[-1]['XFSSprd'])
            
    f=open(out_filename,'wb')
    pickle.dump([results,mult_inf_range,nrip_range,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd],f)
    f.close()
    
if PlotTheExperiment  :
    
    f=open(out_filename,'rb')
    [results,mult_inf_range,nrip_range,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
    f.close()
    
    import matplotlib.pyplot as plt 

    plt.pcolormesh(nrip_range,mult_inf_range,total_analysis_rmse)
    plt.colorbar()
    plt.title('Analysis Rmse')
    plt.xlabel('Rip Iterantions')
    plt.ylabel('Multiplicative Inflation')
    plt.show()

    plt.plot(total_analysis_sprd[:,0],total_analysis_rmse[:,0]);plt.plot(total_analysis_sprd[:,1],total_analysis_rmse[:,1]);plt.plot(total_analysis_sprd[:,-1],total_analysis_rmse[:,-1])

    plt.show()


