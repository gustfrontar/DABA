#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:50:17 2020
@author: jruiz
"""

import pickle
import sys
import os
sys.path.append('../model/')
sys.path.append('../data_assimilation/')

import numpy as np
import sensitivity_conf_default as conf
import assimilation_letkf_module as alm

Force=False #When false we will check if the output exist before running the experiment again.
NatureName = sys.argv[1]

conf.GeneralConf['NatureName']=NatureName
out_filename='./npz/Sesitivity_experiment_multinfyloc_LETKF_ptemp2.0_' + conf.GeneralConf['NatureName'] + '.npz'
#Define the source of the observations
conf.GeneralConf['ObsFile']='./data/Nature/'+conf.GeneralConf['NatureName']+'.npz'
    
conf.DAConf['ExpLength'] = 250                           #None use the full nature run experiment. Else use this length.
conf.DAConf['NEns'] = 20                                  #Number of ensemble members
conf.DAConf['Twin'] = True                                #When True, model configuration will be replaced by the model configuration in the nature run.
conf.DAConf['BridgeParam']=0.0                            #Bridging parameter for the hybrid 0-pure LETKF, 1.0-pure ETPF
conf.DAConf['AddaptiveTemp']=False                        #Enable addaptive tempering time step in pseudo time.
conf.DAConf['AlphaTempScale'] = 2.0                       #Scale factor to obtain the tempering factors on each tempering iteration.
conf.DAConf['GrossCheckFactor'] = 1000.0                  #Optimized gross error check
conf.DAConf['LowDbzPerThresh']  = 1.1                     #Optimized Low ref thresh.

results=list()

if ( os.path.exists(out_filename) & ~Force  ) :
   print('Warning: Output file exists, I will skip this experiment')
   print( out_filename )
   quit()
    
mult_inf_range  = np.arange(1.05,1.2,0.1)  #Inflation range
loc_scale_range = np.arange(1.0,1.5,0.5)    #Localization range
temp_range      = np.array([1,2,3])          #N iteration range
AlphaTempList = []

Output = dict() 
    
Output['total_analysis_rmse'] = np.zeros( (len(mult_inf_range),len(loc_scale_range),len(temp_range)) )
Output['total_analysis_sprd'] = np.zeros( (len(mult_inf_range),len(loc_scale_range),len(temp_range)) )
Output['total_forecast_rmse'] = np.zeros( (len(mult_inf_range),len(loc_scale_range),len(temp_range)) )
Output['total_forecast_sprd'] = np.zeros( (len(mult_inf_range),len(loc_scale_range),len(temp_range)) )
Output['total_forecast_bias'] = np.zeros( (len(mult_inf_range),len(loc_scale_range),len(temp_range)) )
Output['total_analysis_bias'] = np.zeros( (len(mult_inf_range),len(loc_scale_range),len(temp_range)) )  
 
for itemp , ntemp in enumerate( temp_range ) : 
   for iinf , mult_inf in enumerate( mult_inf_range ) :
      for iloc , loc_scale in enumerate( loc_scale_range ) :
            
        conf.DAConf['InfCoefs']=np.array([mult_inf,0.0,0.0,0.0,0.0])
        conf.DAConf['LocScalesLETKF'] = np.array([loc_scale,-1.0])
        conf.DAConf['NTemp']=ntemp
            
        results = alm.assimilation_letkf_run( conf ) 
        AlphaTempList.append( alm.get_temp_steps( conf.DAConf['NTemp'] , conf.DAConf['AlphaTempScale'] ) )
                 
        print('Multiplicative Inflation',mult_inf)
        print('Localization Scale',loc_scale)
        print('AlphaTemp',AlphaTempList[-1])
        print('Analisis RMSE: ',np.mean(results['XASRmse']))
        print('Forecast RMSE: ',np.mean(results['XFSRmse']))
        print('Analisis SPRD: ',np.mean(results['XASSprd']))
        print('Forecast SPRD: ',np.mean(results['XFSSprd']))
            
        Output['total_analysis_rmse'][iinf,iloc,itemp] = np.mean(results['XASRmse'])
        Output['total_forecast_rmse'][iinf,iloc,itemp] = np.mean(results['XFSRmse'])
        Output['total_analysis_sprd'][iinf,iloc,itemp] = np.mean(results['XASSprd'])
        Output['total_forecast_sprd'][iinf,iloc,itemp] = np.mean(results['XFSSprd'])
        Output['total_analysis_bias'][iinf,iloc,itemp] = np.mean(results['XASBias'])
        Output['total_forecast_bias'][iinf,iloc,itemp] = np.mean(results['XFSBias'])

        if (itemp == 0 ) & ( iinf == 0 ) & ( iloc == 0) : 
           #This is the first iteration. Save additional output data.
           Output['ObsLoc'] = np.copy( results['ObsLoc'] )
           Output['ObsType'] = np.copy( results['ObsType'] )
           Output['ObsError'] = np.copy( results['ObsError'] )
           Output['YObs'] = np.copy( results['YObs'] )
           Output['XNature'] = np.copy( results['XNature'] )
           DALength = np.shape( results['XNature'] )[1]
           NX = np.shape( results['XNature'] )[0]
           Output['XAMean'] = np.zeros( (len(mult_inf_range),len(loc_scale_range),len(temp_range),NX,DALength) )
           Output['XFMean'] = np.zeros( (len(mult_inf_range),len(loc_scale_range),len(temp_range),NX,DALength) )

        Output['XAMean'][iinf,iloc,itemp,:,:] = np.copy( results['XAMean'] )
        Output['XFMean'][iinf,iloc,itemp,:,:] = np.copy( results['XFMean'] )

        print('XAMean',Output['XAMean'][0,0,0,10,10],Output['XAMean'].shape)

Output['NTempRange'] = temp_range
Output['MultInfRange'] = mult_inf_range
Output['LocScaleRange'] = loc_scale_range
Output['AlphaTempList'] = AlphaTempList           
f=open(out_filename,'wb')
pickle.dump(Output,f)
f.close()
    



