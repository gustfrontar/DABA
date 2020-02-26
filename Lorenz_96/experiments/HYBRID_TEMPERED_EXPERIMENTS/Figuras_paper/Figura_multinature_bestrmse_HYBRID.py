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
#    LEO LOS EXPERIMENTOS DE RIP
#===========================================================================================================

Nature_name=['Logaritmic','Cuadratic','Acevedo','Linear']

file_list=['../../npz/Sesitivity_experiment_bridge_multinf_trip_hybrid_NatureR4_Den05_Freq8_Hlogaritmic.npz',
           '../../npz/Sesitivity_experiment_bridge_multinf_trip_hybrid_NatureR4_Den05_Freq8_Hcuadratic.npz' ,
           '../../npz/Sesitivity_experiment_bridge_multinf_trip_hybrid_NatureR8_Den05_Freq8_Hlinear.npz'    ,
           '../../npz/Sesitivity_experiment_bridge_multinf_trip_hybrid_NatureR1_Den1_Freq4_Hlinear.npz' ]

analysis_rmse_rip=[]
forecast_rmse_rip=[] 
analysis_sprd_rip=[]
forecast_sprd_rip=[]

bridge_range_rip=[]
mult_inf_range_rip=[]
   
for my_file in file_list : 

   f=open(my_file,'rb')
   [results,mult_inf_range,bridge_range,rip_range,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
   f.close()
      
   analysis_rmse_rip.append( total_analysis_rmse )
   analysis_sprd_rip.append( total_analysis_sprd ) 
   forecast_rmse_rip.append( total_forecast_rmse )
   forecast_sprd_rip.append( total_forecast_sprd )

   bridge_range_rip.append( bridge_range )
   mult_inf_range_rip.append( mult_inf_range )
   
#===========================================================================================================
#    LEO LOS EXPERIMENTOS DE ORIP
#===========================================================================================================


file_list=['../../npz/Sesitivity_experiment_bridge_multinf_orip_hybrid_NatureR4_Den05_Freq8_Hlogaritmic.npz',
           '../../npz/Sesitivity_experiment_bridge_multinf_orip_hybrid_NatureR4_Den05_Freq8_Hcuadratic.npz' ,
           '../../npz/Sesitivity_experiment_bridge_multinf_orip_hybrid_NatureR8_Den05_Freq8_Hlinear.npz'    ,
           '../../npz/Sesitivity_experiment_bridge_multinf_orip_hybrid_NatureR1_Den1_Freq4_Hlinear.npz' ]

analysis_rmse_orip=[]
forecast_rmse_orip=[] 
analysis_sprd_orip=[]
forecast_sprd_orip=[]

bridge_range_orip=[]
mult_inf_range_orip=[]
   
for my_file in file_list : 

   f=open(my_file,'rb')
   [results,mult_inf_range,bridge_range,rip_range,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
   f.close()
      
   analysis_rmse_orip.append( total_analysis_rmse )
   analysis_sprd_orip.append( total_analysis_sprd ) 
   forecast_rmse_orip.append( total_forecast_rmse )
   forecast_sprd_orip.append( total_forecast_sprd )

   bridge_range_orip.append( bridge_range )
   mult_inf_range_orip.append( mult_inf_range )  
   
   
   

#===========================================================================================================
#    LEO LOS EXPERIMENTOS DE ADRIP
#===========================================================================================================

file_list=['../../npz/Sesitivity_experiment_bridge_multinf_adrip_hybrid_NatureR4_Den05_Freq8_Hlogaritmic.npz',
           '../../npz/Sesitivity_experiment_bridge_multinf_adrip_hybrid_NatureR4_Den05_Freq8_Hcuadratic.npz',
           '../../npz/Sesitivity_experiment_bridge_multinf_adrip_hybrid_NatureR8_Den05_Freq8_Hlinear.npz'    ,
           '../../npz/Sesitivity_experiment_bridge_multinf_trip_hybrid_NatureR1_Den1_Freq4_Hlinear.npz' ]

analysis_rmse_adrip=[]
forecast_rmse_adrip=[] 
analysis_sprd_adrip=[]
forecast_sprd_adrip=[]

bridge_range_adrip=[]
mult_inf_range_adrip=[]
    
for my_file in file_list : 

   f=open(my_file,'rb')
   [results,mult_inf_range,bridge_range,rip_range,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
   f.close()
   
   analysis_rmse_adrip.append( total_analysis_rmse )
   analysis_sprd_adrip.append( total_analysis_sprd ) 
   forecast_rmse_adrip.append( total_forecast_rmse )
   forecast_sprd_adrip.append( total_forecast_sprd )
   
   bridge_range_adrip.append( bridge_range )
   mult_inf_range_adrip.append( mult_inf_range )
   
#===========================================================================================================
#    LEO LOS EXPERIMENTOS DE ADTEMP
#===========================================================================================================

file_list=['../../npz/Sesitivity_experiment_bridge_multinf_adtemp_hybrid_NatureR4_Den05_Freq8_Hlogaritmic.npz',
           '../../npz/Sesitivity_experiment_bridge_multinf_adtemp_hybrid_NatureR4_Den05_Freq8_Hcuadratic.npz',
           '../../npz/Sesitivity_experiment_bridge_multinf_adtemp_hybrid_NatureR8_Den05_Freq8_Hlinear.npz'    ,
           '../../npz/Sesitivity_experiment_bridge_multinf_adtemp_hybrid_NatureR1_Den1_Freq4_Hlinear.npz' ]

analysis_rmse_adtemp=[]
forecast_rmse_adtemp=[] 
analysis_sprd_adtemp=[]
forecast_sprd_adtemp=[]

bridge_range_adtemp=[]
mult_inf_range_adtemp=[]

    
for my_file in file_list : 

   f=open(my_file,'rb')
   [results,mult_inf_range,bridge_range,temp_range,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
   f.close()
   
   analysis_rmse_adtemp.append( total_analysis_rmse )
   analysis_sprd_adtemp.append( total_analysis_sprd ) 
   forecast_rmse_adtemp.append( total_forecast_rmse )
   forecast_sprd_adtemp.append( total_forecast_sprd )   
   
   bridge_range_adtemp.append( bridge_range )
   mult_inf_range_adtemp.append( mult_inf_range )
   
   
#Make a plot for each nature showing the optimal combination of inflation and bridge parameter.
 
for inat , nat in enumerate( Nature_name ) :   
   
   #Figure with trip.
   analysis_rmse = analysis_rmse_rip[inat]
   bridge_range  = bridge_range_rip[inat]
   mult_inf_range = mult_inf_range_rip[inat]
   plt.figure()
          
   rmse_max = np.nanmax( analysis_rmse )
   rmse_min = np.nanmin( analysis_rmse )
   drmse = ( rmse_max - rmse_min ) / 20.0
   levels = np.arange( rmse_min , rmse_max + drmse , drmse )
   
   for iiter in range( np.size(rip_range) )  :
       plt.subplot(1,3,iiter+1)
       plt.contourf(bridge_range,mult_inf_range,analysis_rmse[:,:,iiter],levels=levels)
       plt.title('N. TRIP iter ='+str(iiter+1) )
       plt.grid()
       
   plt.savefig('Figure_HIBRID_bridge_multinf_trip_'+nat+'.png')   

for inat , nat in enumerate( Nature_name ) :   
   
   #Figure with trip.
   analysis_rmse = analysis_rmse_orip[inat]
   bridge_range  = bridge_range_orip[inat]
   mult_inf_range = mult_inf_range_orip[inat]
   plt.figure()
          
   rmse_max = np.nanmax( analysis_rmse )
   rmse_min = np.nanmin( analysis_rmse )
   drmse = ( rmse_max - rmse_min ) / 20.0
   levels = np.arange( rmse_min , rmse_max + drmse , drmse )
   
   for iiter in range( np.size(rip_range) )  :
       plt.subplot(1,3,iiter+1)
       plt.contourf(bridge_range,mult_inf_range,analysis_rmse[:,:,iiter],levels=levels)
       plt.title('N. ORIP iter ='+str(iiter+1) )
       plt.grid()    
   plt.savefig('Figure_HIBRID_bridge_multinf_orip_'+nat+'.png')  
       
for inat , nat in enumerate( Nature_name ) :   
   
   #Figure with trip.
   analysis_rmse = analysis_rmse_adrip[inat] 
   bridge_range  = bridge_range_adrip[inat]
   mult_inf_range = mult_inf_range_adrip[inat]
   plt.figure()
          
   rmse_max = np.nanmax( analysis_rmse )
   rmse_min = np.nanmin( analysis_rmse )
   drmse = ( rmse_max - rmse_min ) / 20.0
   levels = np.arange( rmse_min , rmse_max + drmse , drmse )
   
   for iiter in range( np.size(rip_range) )  :
       plt.subplot(1,3,iiter+1)
       plt.contourf(bridge_range,mult_inf_range,analysis_rmse[:,:,iiter],levels=levels)
       plt.title('N. TRIP iter ='+str(iiter+1) )
       plt.grid()    
   plt.savefig('Figure_HIBRID_bridge_multinf_adtrip_'+nat+'.png')  
   
for inat , nat in enumerate( Nature_name ) :   
   
   #Figure with trip.
   analysis_rmse = analysis_rmse_adtemp[inat]
   bridge_range  = bridge_range_adtemp[inat]
   mult_inf_range = mult_inf_range_adtemp[inat]
   plt.figure()
          
   rmse_max = np.nanmax( analysis_rmse )
   rmse_min = np.nanmin( analysis_rmse )
   drmse = ( rmse_max - rmse_min ) / 20.0
   levels = np.arange( rmse_min , rmse_max + drmse , drmse )
   
   for iiter in range( np.size(temp_range) )  :
       plt.subplot(1,3,iiter+1)
       plt.contourf(bridge_range,mult_inf_range,analysis_rmse[:,:,iiter],levels=levels)
       plt.title('N. TEMP iter ='+str(iiter+1) )
       plt.grid()    
   plt.savefig('Figure_HIBRID_bridge_multinf_adtemp_'+nat+'.png')     
   
   


#For each nature plot the minimium rmse as a function of TRIP iteration (including trip and adtrip)  
#And compare the hibrid and the LETKF.
   
plt.figure()  

for inat , nat in enumerate( Nature_name ) :   
   plt.subplot(2,2,inat+1)
   
   #Figure with trip.
   rmse_trip = analysis_rmse_rip[inat] 
   rmse_orip   = analysis_rmse_orip[inat]
   rmse_adtrip = analysis_rmse_adrip[inat]
   rmse_adtemp = analysis_rmse_adtemp[inat]

   
   rmse_min_trip_etkf=np.zeros(np.size(rip_range))
   rmse_min_trip_hib =np.zeros(np.size(rip_range))
   rmse_min_orip_etkf=np.zeros(np.size(rip_range))
   rmse_min_orip_hib =np.zeros(np.size(rip_range))
   rmse_min_adtrip_etkf=np.zeros(np.size(rip_range))
   rmse_min_adtrip_hib =np.zeros(np.size(rip_range))
   rmse_min_adtemp_etkf=np.zeros(np.size(temp_range))
   rmse_min_adtemp_hib =np.zeros(np.size(temp_range))
   
   for iiter in range( np.size(rip_range) ) :
       rmse_min_trip_etkf[iiter]    = np.nanmin( rmse_trip[:,0,iiter] )  #Tomo el minimio considerando solo el bridge parameter =0.
       rmse_min_trip_hib[iiter]     = np.nanmin( rmse_trip[:,1:,iiter])  #Tomo el minimo sobre todos los valores de bridge parameter excepto el 0.
       rmse_min_orip_etkf[iiter]    = np.nanmin( rmse_orip[:,0,iiter] )  #Tomo el minimio considerando solo el bridge parameter =0.
       rmse_min_orip_hib[iiter]     = np.nanmin( rmse_orip[:,1:,iiter])  #Tomo el minimo sobre todos los valores de bridge parameter excepto el 0.
       rmse_min_adtrip_etkf[iiter]  = np.nanmin( rmse_adtrip[:,0,iiter] )  #Tomo el minimio considerando solo el bridge parameter =0.
       rmse_min_adtrip_hib[iiter]   = np.nanmin( rmse_adtrip[:,1:,iiter])  #Tomo el minimo sobre todos los valores de bridge parameter excepto el 0.   
       rmse_min_adtemp_etkf[iiter]  = np.nanmin( rmse_adtemp[:,0,iiter] )  #Tomo el minimio considerando solo el bridge parameter =0.
       rmse_min_adtemp_hib[iiter]   = np.nanmin( rmse_adtemp[:,1:,iiter])  #Tomo el minimo sobre todos los valores de bridge parameter excepto el 0.   
       
           
   #plt.plot(rip_range,rmse_min_trip_etkf,'r-')
   #plt.plot(rip_range,rmse_min_trip_hib ,'b-')
   plt.plot(rip_range,rmse_min_orip_etkf,'rs')
   plt.plot(rip_range,rmse_min_orip_hib,'bs')
   plt.plot(rip_range,rmse_min_adtrip_etkf,'r--')
   plt.plot(rip_range,rmse_min_adtrip_hib,'b--')
   plt.plot(rip_range,rmse_min_adtemp_etkf,'r.')
   plt.plot(rip_range,rmse_min_adtemp_hib,'b.')
   plt.xlabel('Iteration')
   plt.ylabel('RMSE')
   plt.title(nat)
   plt.grid()
   
plt.savefig('Figure_HIBRID_bridge_multinf_minrmse_'+nat+'.png') 


#Una figura que muestre como depende el RMSE del spread del ensamble para un beta fijo de 0.2 y para el LETKF puro



plt.figure()

for inat , nat in enumerate( Nature_name ) :   
   plt.subplot(2,2,inat+1)
   

   plt.plot( analysis_sprd_adrip[inat][:,0,0] , analysis_rmse_adrip[inat][:,0,0] ,'r-')
   plt.plot( analysis_sprd_adrip[inat][:,0,1] , analysis_rmse_adrip[inat][:,0,1] ,'r--')

   plt.plot( analysis_sprd_adrip[inat][:,2,0] , analysis_rmse_adrip[inat][:,2,0] ,'b-')
   plt.plot( analysis_sprd_adrip[inat][:,2,1] , analysis_rmse_adrip[inat][:,2,1] ,'b--')

   plt.plot( analysis_sprd_orip[inat][:,0,1] , analysis_rmse_orip[inat][:,0,1] ,'rs')
   plt.plot( analysis_sprd_orip[inat][:,2,1] , analysis_rmse_orip[inat][:,2,1] ,'bs')
   plt.grid() 
      
plt.savefig('Figure_HIBRID_bridge_multinf_adtemp_spreadvsrmse_'+nat+'.png') 
   