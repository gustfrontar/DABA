import matplotlib.pyplot as plt
import pickle
import numpy as np

plt.rcParams.update({'font.size': 15})


NatureName='NatureR1_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_tempering_multinf_F7.5_LETKF_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,inf_range,AlphaTempList,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

Nobs = np.zeros( total_analysis_rmse.shape )

idx = 0
for iinf , mult_inf in enumerate( inf_range ) :
  for intemp in range( Nobs.shape[1] ) :
     Nobs[iinf,intemp] = np.mean( results[idx]['Nobs'] )
     idx=idx+1
     
fig , axs = plt.subplots( 1 , 2 , figsize=(12,5) , sharey = True )

max_obs = np.round( np.max( Nobs , axis = 0 ) , 2 )

p1=axs[0].plot(inf_range,Nobs[:,0],label='EnKF ('+str(max_obs[0])+')')
p2=axs[0].plot(inf_range,Nobs[:,1],label='EnKF-T2 ('+str(max_obs[1])+')')
p3=axs[0].plot(inf_range,Nobs[:,2],label='EnKF-T3 ('+str(max_obs[2])+')')
p4=axs[0].plot(inf_range,Nobs[:,3],label='EnKF-T4 ('+str(max_obs[3])+')')
axs[0].legend()
axs[0].grid()
#axs[0].set_ylim([0,5])
axs[0].set_xlabel('Inflation')
axs[0].set_ylabel('# of Assimilated Obs.')
axs[0].set_title('(a)')

NatureName='NatureR5_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_tempering_multinf_F7.5_LETKF_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,inf_range,AlphaTempList,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

Nobs = np.zeros( total_analysis_rmse.shape )

idx = 0
for iinf , mult_inf in enumerate( inf_range ) :
  for intemp in range( Nobs.shape[1] ) :
     Nobs[iinf,intemp] = np.mean( results[idx]['Nobs'] )
     idx=idx+1 

max_obs = np.round( np.max( Nobs , axis = 0 ) , 2 )

p1=axs[1].plot(inf_range,Nobs[:,0],label='EnKF ('+str(max_obs[0])+')')
p2=axs[1].plot(inf_range,Nobs[:,1],label='EnKF-T2 ('+str(max_obs[1])+')')
p3=axs[1].plot(inf_range,Nobs[:,2],label='EnKF-T3 ('+str(max_obs[2])+')')
p4=axs[1].plot(inf_range,Nobs[:,3],label='EnKF-T4 ('+str(max_obs[3])+')')
axs[1].legend()
axs[1].grid()
#axs[1].set_ylim([0,5])
axs[1].set_xlabel('Inflation')
axs[1].set_ylabel('# of Assimilated Obs.')
axs[1].set_title('(b)')

plt.savefig('FigureNobs_Temp_Multinf_F7.5_Den1_Freq4_Hradar.png')
