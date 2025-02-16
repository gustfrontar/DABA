import matplotlib.pyplot as plt
import pickle
import numpy as np

plt.rcParams.update({'font.size': 15})


NatureName='NatureR1_Den1_Freq4_Hquadratic'
exp_filename='../npz/Sesitivity_experiment_ptemp2.0_multinf_LETKF_' + NatureName + '.npz'



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

fig , axs = plt.subplots( 1 , 3 , figsize=(15,5) , sharey = True )

min_error = np.round( np.nanmin( total_analysis_rmse , axis = 0 ) , 2 )

axs[0].plot( inf_range , total_analysis_rmse[:,0] , label= 'EnKF ('+str(min_error[0])+')' )
axs[0].plot(inf_range,total_analysis_rmse[:,1],label='EnKF-T2 ('+str(min_error[1])+')')
axs[0].plot(inf_range,total_analysis_rmse[:,2],label='EnKF-T3 ('+str(min_error[2])+')')
axs[0].plot(inf_range,total_analysis_rmse[:,3],label='EnKF-T4 ('+str(min_error[3])+')')
axs[0].legend()
axs[0].grid()
axs[0].set_ylim([0,10])
axs[0].set_xlabel('Inflation')
axs[0].set_ylabel('Analysis RMSE')
axs[0].set_title('(a)')

NatureName='NatureR5_Den1_Freq4_Hquadratic'

exp_filename='../npz/Sesitivity_experiment_ptemp2.0_multinf_LETKF_' + NatureName + '.npz'

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

min_error = np.round( np.nanmin( total_analysis_rmse , axis = 0 ) , 2 )

p1=axs[1].plot(inf_range,total_analysis_rmse[:,0],label='EnKF ('+str(min_error[0])+')')
p2=axs[1].plot(inf_range,total_analysis_rmse[:,1],label='EnKF-T2 ('+str(min_error[1])+')')
p3=axs[1].plot(inf_range,total_analysis_rmse[:,2],label='EnKF-T3 ('+str(min_error[2])+')')
p4=axs[1].plot(inf_range,total_analysis_rmse[:,3],label='EnKF-T4 ('+str(min_error[3])+')')
axs[1].legend()
axs[1].grid()
axs[1].set_ylim([0,10])
axs[1].set_xlabel('Inflation')
#axs[1].set_ylabel('Analysis RMSE')
axs[1].set_title('(b)')

NatureName='NatureR25_Den1_Freq4_Hquadratic'

exp_filename='../npz/Sesitivity_experiment_ptemp2.0_multinf_LETKF_' + NatureName + '.npz'

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

min_error = np.round( np.nanmin( total_analysis_rmse , axis = 0 ) , 2 )

p1=axs[2].plot(inf_range,total_analysis_rmse[:,0],label='EnKF ('+str(min_error[0])+')')
p2=axs[2].plot(inf_range,total_analysis_rmse[:,1],label='EnKF-T2 ('+str(min_error[1])+')')
p3=axs[2].plot(inf_range,total_analysis_rmse[:,2],label='EnKF-T3 ('+str(min_error[2])+')')
p4=axs[2].plot(inf_range,total_analysis_rmse[:,3],label='EnKF-T4 ('+str(min_error[3])+')')
axs[2].legend()
axs[2].grid()
axs[2].set_ylim([0,5])
axs[2].set_xlabel('Inflation')
#axs[1].set_ylabel('Analysis RMSE')
axs[2].set_title('(c)')


plt.savefig('FigureAnalRMSE_Temp_Multinf_Den1_Freq4_Hquadratic.png')
