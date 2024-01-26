import matplotlib.pyplot as plt
import pickle
import numpy as np

plt.rcParams.update({'font.size': 15})


NatureName='NatureR1_Den1_Freq4_Hradar'


exp_filename='../npz/Sesitivity_experiment_ptemp2.0_multinf_LETKF_' + NatureName + '_ng.npz'

f=open(exp_filename,'rb')
[results,inf_range,AlphaTempList,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( total_analysis_rmse.shape )
for ii in range( total_analysis_rmse.shape[0] ) :
    for jj in range( total_analysis_rmse.shape[1] ) :
        NormalEnd[ii,jj] = 1-results[ii*total_analysis_rmse.shape[1]+jj]['NormalEnd']
NormalEnd=NormalEnd.astype(bool)  
total_analysis_rmse[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan  


fig , axs = plt.subplots( 1 , 1 , figsize=(5,6) , sharey = True )

min_error = np.round( np.nanmin( total_analysis_rmse , axis = 0 ) , 2 )

axs.plot( inf_range , total_analysis_rmse[:,0] , label= 'LETKF ('+str(min_error[0])+')' )
axs.plot(inf_range,total_analysis_rmse[:,1],label='LETKF-T2 ('+str(min_error[1])+')')
axs.plot(inf_range,total_analysis_rmse[:,2],label='LETKF-T3 ('+str(min_error[2])+')')
axs.plot(inf_range,total_analysis_rmse[:,3],label='LETKF-T4 ('+str(min_error[3])+')')
axs.grid()
axs.set_ylim([0,5])
axs.set_xlabel('Inflation')
axs.set_ylabel('Analysis RMSE')
axs.set_title('(a)')

NatureName='NatureR1_Den1_Freq4_Hlinear'
exp_filename='../npz/Sesitivity_experiment_ptemp2.0_multinf_LETKF_' + NatureName + '.npz'
f=open(exp_filename,'rb')
[results,inf_range,AlphaTempList,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()
min_error = np.round( np.min( total_analysis_rmse , axis = 0 ) , 2 )
axs.plot( inf_range , total_analysis_rmse[:,0] ,'k--', label= 'Linear EnKF ('+str(min_error[0])+')' )
axs.legend(loc='upper right')



plt.savefig('FigureAnalRMSE_Temp_Multinf_R1_Den1_Freq4_Hradar_ng.png')
