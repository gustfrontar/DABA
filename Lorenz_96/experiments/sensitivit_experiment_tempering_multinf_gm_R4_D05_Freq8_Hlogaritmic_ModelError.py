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
import assimilation_gm_module as ahm

if len(sys.argv) > 1 and sys.argv[1] == 'compute' :
   RunTheExperiment = True
   PlotTheExperiment = False
else                        :
   RunTheExperiment = False
   PlotTheExperiment = True


conf.GeneralConf['NatureName']='NatureR4_Den05_Freq8_Hlogaritmic'
out_filename='Sesitivity_experiment_temp_multif_GM_' + conf.GeneralConf['NatureName'] + '_ModelError.npz'
#Define the source of the observations
conf.GeneralConf['ObsFile']='./data/Nature/'+conf.GeneralConf['NatureName']+'.npz'
    
conf.DAConf['ExpLength'] = None                           #None use the full nature run experiment. Else use this length.
conf.DAConf['NEns'] = 20                                  #Number of ensemble members
conf.DAConf['Freq'] = 8                                   #Assimilation frequency (in number of time steps)
conf.DAConf['TSFreq'] = 8                                 #Intra window ensemble output frequency (for 4D Data assimilation)
conf.DAConf['LocScalesLETKF']=np.array([3.0,-1.0])        #Localization scale is space and time (negative means no localization)
conf.DAConf['LocScalesLETPF']=np.array([3.0,-1.0])        #Localization scale is space and time (negative means no localization)

conf.DAConf['BetaCoef']=0.6                                 #Scaling parameter for the Gaussian Kernel in the Gaussian mixture prior.
conf.DAConf['GammaCoef']=0.2                                #Nudging parameter to uniform weigths in order to avoid weigth degeneracy. 
conf.DAConf['ResamplingType']=2                             #Resampling: 1-Liu 2016, 2-Acevedo et al. 2016, 3-NETPF without rotation, 4-NETPF with rotation.

#Introduce a model error in the model used for the assimilation experiment.
conf.DAConf['Twin'] = False                               #When True, model configuration will be replaced by the model configuration in the nature run.
conf.ModelConf['Coef']=np.array([7.0])                    #Coefficient of parametrized forcing (polynom coefficients starting from coef[0]*x^0 + coef[1]*x ... )


if RunTheExperiment  :

    
    results=list()
    
    mult_inf_range = np.arange(1.15,1.35,0.02)
    
    ntemp_range = np.arange(1,5,1)
    
    total_analysis_rmse = np.zeros( (len(mult_inf_range),len(ntemp_range)) )
    total_analysis_sprd = np.zeros( (len(mult_inf_range),len(ntemp_range)) )
    total_forecast_rmse = np.zeros( (len(mult_inf_range),len(ntemp_range)) )
    total_forecast_sprd = np.zeros( (len(mult_inf_range),len(ntemp_range)) )
    
    for iinf , mult_inf in enumerate( mult_inf_range ) :
        for intemp , ntemp in enumerate( ntemp_range )  :
            
            conf.DAConf['InfCoefs']=np.array([mult_inf,0.0,0.0,0.0,0.0])
            conf.DAConf['NTemp']=int(ntemp) 
            
            results.append( ahm.assimilation_gm_run( conf ) )
                 
            print('Multiplicative Inflation',mult_inf)
            print('Tempering iteraations',ntemp)
            print('Analisis RMSE: ',np.mean(results[-1]['XASRmse']))
            print('Forecast RMSE: ',np.mean(results[-1]['XFSRmse']))
            print('Analisis SPRD: ',np.mean(results[-1]['XASSprd']))
            print('Forecast SPRD: ',np.mean(results[-1]['XFSSprd']))
            
            total_analysis_rmse[iinf,intemp] = np.mean(results[-1]['XASRmse'])
            total_forecast_rmse[iinf,intemp] = np.mean(results[-1]['XFSRmse'])
            total_analysis_sprd[iinf,intemp] = np.mean(results[-1]['XASSprd'])
            total_forecast_sprd[iinf,intemp] = np.mean(results[-1]['XFSSprd'])
            
    f=open(out_filename,'wb')
    pickle.dump([results,mult_inf_range,ntemp_range,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd],f)
    f.close()
    
if PlotTheExperiment  :
    
    f=open(out_filename,'rb')
    [results,mult_inf_range,ntemp_range,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
    f.close()
    
    import matplotlib.pyplot as plt 

    plt.pcolormesh(ntemp_range,mult_inf_range,total_analysis_rmse)
    plt.colorbar()
    plt.title('Analysis Rmse')
    plt.xlabel('Tempering Iterantions')
    plt.ylabel('Multiplicative Inflation')
    plt.show()

    plt.plot(total_analysis_sprd[:,0],total_analysis_rmse[:,0]);plt.plot(total_analysis_sprd[:,1],total_analysis_rmse[:,1]);plt.plot(total_analysis_sprd[:,-1],total_analysis_rmse[:,-1])


