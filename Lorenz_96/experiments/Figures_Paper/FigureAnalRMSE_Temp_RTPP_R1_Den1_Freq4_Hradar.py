import matplotlib.pyplot as plt
import pickle
import numpy as np

plt.rcParams.update({'font.size': 15})


NatureName='NatureR1_Den1_Freq4_Hradar'


exp_filename='../npz/Sesitivity_experiment_ptemp2.0_rtps_LETKF_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,inf_range,AlphaTempList,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

fig , axs = plt.subplots( 1 , 2 , figsize=(12,5) , sharey = True )

min_error = np.round( np.min( total_analysis_rmse , axis = 0 ) , 2 )

axs[0].plot( inf_range , total_analysis_rmse[:,0] , label= 'EnKF ('+str(min_error[0])+')' )
axs[0].plot(inf_range,total_analysis_rmse[:,1],label='EnKF-T2 ('+str(min_error[1])+')')
axs[0].plot(inf_range,total_analysis_rmse[:,2],label='EnKF-T3 ('+str(min_error[2])+')')
axs[0].plot(inf_range,total_analysis_rmse[:,3],label='EnKF-T4 ('+str(min_error[3])+')')
axs[0].legend()
axs[0].grid()
axs[0].set_ylim([0,5])
axs[0].set_xlabel('Inflation')
axs[0].set_ylabel('Analysis RMSE')
axs[0].set_title('(a)')


NatureName='NatureR5_Den1_Freq4_Hradar'

exp_filename='../npz/Sesitivity_experiment_ptemp2.0_rtps_LETKF_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,inf_range,AlphaTempList,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

min_error = np.round( np.min( total_analysis_rmse , axis = 0 ) , 2 )

p1=axs[1].plot(inf_range,total_analysis_rmse[:,0],label='EnKF ('+str(min_error[0])+')')
p2=axs[1].plot(inf_range,total_analysis_rmse[:,1],label='EnKF-T2 ('+str(min_error[1])+')')
p3=axs[1].plot(inf_range,total_analysis_rmse[:,2],label='EnKF-T3 ('+str(min_error[2])+')')
p4=axs[1].plot(inf_range,total_analysis_rmse[:,3],label='EnKF-T4 ('+str(min_error[3])+')')
axs[1].legend()
axs[1].grid()
axs[1].set_ylim([0,5])
axs[1].set_xlabel('Inflation')
axs[1].set_ylabel('Analysis RMSE')
axs[1].set_title('(b)')


plt.savefig('FigureAnalRMSE_Temp_RTPP_R1_Den1_Freq4_Hradar.png')
