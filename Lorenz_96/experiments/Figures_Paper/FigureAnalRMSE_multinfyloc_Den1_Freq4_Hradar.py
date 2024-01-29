import matplotlib.pyplot as plt
import pickle
import numpy as np
plt.rcParams['text.usetex'] = True

plt.rcParams.update({'font.size': 22})

NatureName='NatureR03_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
NormalEnd=NormalEnd.astype(bool)    
        
total_analysis_rmse[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan      

fig , axs = plt.subplots( 4 , 1 , figsize=(7,24) )

pcolor0=axs[0].pcolor( loc_scale_range , mult_inf_range ,  total_analysis_rmse , vmin=0 , vmax=5.0 , cmap='YlGn' )
plt.colorbar(pcolor0,ax=axs[0])
axs[0].set_ylabel('Mult. Inf.')

axs[0].set_title('(a)')


NatureName='NatureR1_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
NormalEnd=NormalEnd.astype(bool)    
        
total_analysis_rmse[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan      

pcolor0=axs[1].pcolor( loc_scale_range , mult_inf_range ,  total_analysis_rmse , vmin=0 , vmax=5.0 , cmap='YlGn' )
plt.colorbar(pcolor0,ax=axs[1])
axs[1].set_ylabel('Mult. Inf.')

axs[1].set_title('(b)')


NatureName='NatureR5_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
NormalEnd=NormalEnd.astype(bool)    
        
total_analysis_rmse[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan      

pcolor0=axs[2].pcolor( loc_scale_range , mult_inf_range ,  total_analysis_rmse , vmin=0 , vmax=5.0 , cmap='YlGn' )
plt.colorbar(pcolor0,ax=axs[2])
axs[2].set_ylabel('Mult. Inf.')

axs[2].set_title('(b)')


NatureName='NatureR25_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
NormalEnd=NormalEnd.astype(bool)    
        
total_analysis_rmse[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan      

pcolor0=axs[3].pcolor( loc_scale_range , mult_inf_range ,  total_analysis_rmse , vmin=0 , vmax=5.0 , cmap='YlGn' )
plt.colorbar(pcolor0,ax=axs[3])
axs[3].set_ylabel('Mult. Inf.')

axs[3].set_title('(b)')




plt.savefig('FigureAnalRMSE_multinfyloc_Den1_Freq4_Hradar.png')
