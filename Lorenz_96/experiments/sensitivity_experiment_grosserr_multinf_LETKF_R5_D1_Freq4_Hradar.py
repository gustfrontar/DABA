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

if len(sys.argv) > 1 and sys.argv[1] == 'compute' :
   RunTheExperiment = True
   PlotTheExperiment = False
else                        :
   RunTheExperiment = False
   PlotTheExperiment = True


conf.GeneralConf['NatureName']='NatureR5_Den1_Freq4_Hradar'
out_filename='./npz/Sesitivity_experiment_tempering_grosserr_LETKF_' + conf.GeneralConf['NatureName'] + '.npz'
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
conf.DAConf['InfCoefs']=np.array([1.41,0.0,0.0,0.0,0.0,0.0,0.0])  #Optimized multiplicative inflation for 1 tempering iteration.
conf.DAConf['NTemp']=1                                    #Single tempering iteration.
conf.DAConf['AlphaTemp'] = np.array([1])
conf.DAConf['AddaptiveTemp']=False                        #Enable addaptive tempering time step in pseudo time.

if RunTheExperiment  :

    results=list()

    gross_error_check_range = np.arange( 1.0 , 9.0 , 0.2 )
    min_dbz_thresh_range    = np.arange( 0.1 , 1.3 , 0.2 )
    #gross_error_check_range = np.array([11.0])
    #min_dbz_thresh_range    = np.array([0.7])
    
    total_analysis_rmse = np.zeros( (len(gross_error_check_range), len(min_dbz_thresh_range) ) )
    total_analysis_sprd = np.zeros( (len(gross_error_check_range), len(min_dbz_thresh_range) ) )
    total_forecast_rmse = np.zeros( (len(gross_error_check_range), len(min_dbz_thresh_range) ) )
    total_forecast_sprd = np.zeros( (len(gross_error_check_range), len(min_dbz_thresh_range) ) )
    
    for igrosse , grosse in enumerate( gross_error_check_range ) :
        for imdbzthr , mindbzthr in enumerate( min_dbz_thresh_range )  :

            conf.DAConf['GrossCheckFactor'] = grosse
            conf.DAConf['LowDbzPerThresh']  = mindbzthr
            
            results.append( ahm.assimilation_hybrid_run( conf ) )
                 
            print('Gross errror check',grosse)
            print('Min. ref. threshold',mindbzthr)
            print('Analisis RMSE: ',np.mean(results[-1]['XASRmse']))
            print('Forecast RMSE: ',np.mean(results[-1]['XFSRmse']))
            print('Analisis SPRD: ',np.mean(results[-1]['XASSprd']))
            print('Forecast SPRD: ',np.mean(results[-1]['XFSSprd']))
            
            total_analysis_rmse[igrosse,imdbzthr] = np.mean(results[-1]['XASRmse'])
            total_forecast_rmse[igrosse,imdbzthr] = np.mean(results[-1]['XFSRmse'])
            total_analysis_sprd[igrosse,imdbzthr] = np.mean(results[-1]['XASSprd'])
            total_forecast_sprd[igrosse,imdbzthr] = np.mean(results[-1]['XFSSprd'])
            
    f=open(out_filename,'wb')
    pickle.dump([results,gross_error_check_range,min_dbz_thresh_range,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd],f)
    f.close()
    
if PlotTheExperiment  :
    
    f=open(out_filename,'rb')
    [results,gross_error_check_range,min_dbz_thresh_range,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
    f.close()
    
    import matplotlib.pyplot as plt 

    plt.figure()
    plt.pcolormesh(gross_error_check_range,min_dbz_thresh_range,total_analysis_rmse)
    plt.colorbar()
    plt.title('Analysis Rmse')
    plt.xlabel('Gross error threshold')
    plt.ylabel('Min ref. threshold')
    plt.show()

    plt.figure()
    plt.plot(total_analysis_sprd[:,0],total_analysis_rmse[:,0]);plt.plot(total_analysis_sprd[:,1],total_analysis_rmse[:,1]);plt.plot(total_analysis_sprd[:,-1],total_analysis_rmse[:,-1])

    plt.show()

