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
import matplotlib.pyplot as plt

import numpy as np
#import sensitivity_conf_default as conf
#import assimilation_ripgm_module as ahm

#===========================================================================================================
#    LEO LOS EXPERIMENTOS DE TEMPERING
#===========================================================================================================

file_list=['../../npz/Sesitivity_experiment_temp_multif_gm_NatureR4_Den05_Freq8_Hcuadratic.npz',
           '../../npz/Sesitivity_experiment_temp_multif_gm_NatureR4_Den05_Freq8_Hlogaritmic.npz',
           '../../npz/Sesitivity_experiment_temp_multif_gm_NatureR8_Den05_Freq8_Hlinear.npz']

analysis_rmse_temp=[]
forecast_rmse_temp=[] 
analysis_sprd_temp=[]
forecast_sprd_temp=[]
    
for my_file in file_list : 

   f=open(my_file,'rb')
   [results,mult_inf_range,nrip_range,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
   f.close()
   
   analysis_rmse_temp.append( total_analysis_rmse )
   analysis_sprd_temp.append( total_analysis_sprd ) 
   forecast_rmse_temp.append( total_forecast_rmse )
   forecast_sprd_temp.append( total_forecast_sprd )

#===========================================================================================================
#    LEO LOS EXPERIMENTOS DE ORIP
#===========================================================================================================

file_list=['../../npz/Sesitivity_experiment_adtemp_multif_gm_NatureR4_Den05_Freq8_Hcuadratic.npz',
           '../../npz/Sesitivity_experiment_adtemp_multif_gm_NatureR4_Den05_Freq8_Hlogaritmic.npz',
           '../../npz/Sesitivity_experiment_adtemp_multif_gm_NatureR8_Den05_Freq8_Hlinear.npz']

analysis_rmse_adtemp=[]
forecast_rmse_adtemp=[] 
analysis_sprd_adtemp=[]
forecast_sprd_adtemp=[]
    
for my_file in file_list : 

   f=open(my_file,'rb')
   [results,mult_inf_range,nrip_range,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
   f.close()
   
   analysis_rmse_adtemp.append( total_analysis_rmse )
   analysis_sprd_adtemp.append( total_analysis_sprd ) 
   forecast_rmse_adtemp.append( total_forecast_rmse )
   forecast_sprd_adtemp.append( total_forecast_sprd )


plt.figure()

plt.subplot(2,2,1)

plt.plot(  np.min( analysis_rmse_temp[0],0)  ,'b-')
plt.plot(  np.min( analysis_rmse_adtemp[0],0) ,'r-')

plt.plot(  np.min( forecast_rmse_temp[0],0)  ,'b--')
plt.plot(  np.min( forecast_rmse_adtemp[0],0) ,'r--')

plt.grid(axis='both')

plt.ylabel('RMSE')
plt.title('Cuadratic operator')

plt.subplot(2,2,2)

plt.plot(  np.min( analysis_rmse_temp[1],0)  ,'b-')
plt.plot(  np.min( analysis_rmse_adtemp[1],0) ,'r-')

plt.plot(  np.min( forecast_rmse_temp[1],0)  ,'b--')
plt.plot(  np.min( forecast_rmse_adtemp[1],0) ,'r--')

#plt.ylim(bottom=0.6,top=2.4)
#plt.xlim(left=0.5,right=0.67)
plt.grid(axis='both')

plt.ylabel('RMSE')
plt.title('Logaritmic operator')

plt.subplot(2,2,3)

plt.plot(  np.min( analysis_rmse_temp[2],0)  ,'b-')
plt.plot(  np.min( analysis_rmse_adtemp[2],0) ,'r-')

plt.plot(  np.min( forecast_rmse_temp[2],0)  ,'b--')
plt.plot(  np.min( forecast_rmse_adtemp[2],0) ,'r--')

#plt.ylim(bottom=1.5,top=1.7)
#plt.xlim(left=0.7,right=1.7)
plt.grid(axis='both')

plt.ylabel('RMSE')
plt.title('Linear operator')


plt.savefig('./Figura_sensibilidad_tempering_minrmse_GM.png')

    # import matplotlib.pyplot as plt 

    # plt.pcolormesh(nrip_range,mult_inf_range,total_analysis_rmse)
    # plt.colorbar()
    # plt.title('Analysis Rmse')
    # plt.xlabel('Rip Iterantions')
    # plt.ylabel('Multiplicative Inflation')
    # plt.show()

    # plt.plot(total_analysis_sprd[:,0],total_analysis_rmse[:,0]);plt.plot(total_analysis_sprd[:,1],total_analysis_rmse[:,1]);plt.plot(total_analysis_sprd[:,-1],total_analysis_rmse[:,-1])


    # plt.show()


