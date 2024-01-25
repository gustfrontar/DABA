import matplotlib.pyplot as plt
import pickle
import numpy as np

plt.rcParams.update({'font.size': 15})




temp_slopes = ['0.0','1.0','2.0','3.0']

ana_rmse = list()
for_rmse = list()
ana_sprd = list()
for_sprd = list()
min_ana_error = list()

NatureName='NatureR1_Den1_Freq4_Hradar'

for islope , slope in enumerate(temp_slopes) : 
   exp_filename='../npz/Sesitivity_experiment_ptemp' + slope + '_multinf_LETKF_' + NatureName + '.npz'
   f=open(exp_filename,'rb')
   [results,inf_range,AlphaTempList,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
   f.close()

   ana_rmse.append( total_analysis_rmse )
   for_rmse.append( total_forecast_rmse )
   min_ana_error.append( np.round( np.min( total_analysis_rmse , axis = 0 ) , 2 ) )

fig , axs = plt.subplots( 1 , 3 , figsize=(15,5) , sharey = True )

min_error = np.round( np.min( total_analysis_rmse , axis = 0 ) , 2 )
axs[0].plot( inf_range , (ana_rmse[0])[:,0] , label= 'EnKF ('+str(min_ana_error[0][0])+')' )
for islope , slope in enumerate( temp_slopes ) :
   axs[0].plot(inf_range, ana_rmse[islope][:,1],label='T2 Slope=' + slope + '(' +str(min_ana_error[islope][1])+')')

axs[0].legend(loc='upper right')
axs[0].grid()
axs[0].set_ylim([0,5])
axs[0].set_xlabel('Inflation')
axs[0].set_ylabel('Analysis RMSE')
axs[0].set_title('(a)')


ana_rmse = list()
for_rmse = list()
ana_sprd = list()
for_sprd = list()
min_ana_error = list()

NatureName='NatureR5_Den1_Freq4_Hradar'

for islope , slope in enumerate(temp_slopes) : 
   exp_filename='../npz/Sesitivity_experiment_ptemp' + slope + '_multinf_LETKF_' + NatureName + '.npz'
   f=open(exp_filename,'rb')
   [results,inf_range,AlphaTempList,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
   f.close()

   ana_rmse.append( total_analysis_rmse )
   for_rmse.append( total_forecast_rmse )
   min_ana_error.append( np.round( np.min( total_analysis_rmse , axis = 0 ) , 2 ) )


axs[1].plot( inf_range , (ana_rmse[0])[:,0] , label= 'EnKF ('+str(min_ana_error[0][0])+')' )
for islope , slope in enumerate( temp_slopes ) :
   axs[1].plot(inf_range,ana_rmse[islope][:,1],label='T2 Slope=' + slope + '(' +str(min_ana_error[islope][2])+')')


axs[1].legend()
axs[1].grid()
axs[1].legend(loc='upper right')
axs[1].set_ylim([0,5])
axs[1].set_xlabel('Inflation')
axs[1].set_ylabel('Analysis RMSE')
axs[1].set_title('(b)')


ana_rmse = list()
for_rmse = list()
ana_sprd = list()
for_sprd = list()
min_ana_error = list()

NatureName='NatureR25_Den1_Freq4_Hradar'

for islope , slope in enumerate(temp_slopes) : 
   exp_filename='../npz/Sesitivity_experiment_ptemp' + slope + '_multinf_LETKF_' + NatureName + '.npz'
   f=open(exp_filename,'rb')
   [results,inf_range,AlphaTempList,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
   f.close()

   ana_rmse.append( total_analysis_rmse )
   for_rmse.append( total_forecast_rmse )
   min_ana_error.append( np.round( np.min( total_analysis_rmse , axis = 0 ) , 2 ) )


axs[2].plot( inf_range , (ana_rmse[0])[:,0] , label= 'EnKF ('+str(min_ana_error[0][0])+')' )
for islope , slope in enumerate( temp_slopes ) :
   axs[2].plot(inf_range,ana_rmse[islope][:,1],label='T2 Slope=' + slope + '(' +str(min_ana_error[islope][2])+')')


axs[2].legend()
axs[2].grid()
axs[2].set_ylim([0,5])
axs[2].set_xlabel('Inflation')
axs[2].set_ylabel('Analysis RMSE')
axs[2].set_title('(c)')
axs[2].legend(loc='upper right')


plt.savefig('FigureAnalRMSE_SensTempSlope_Temp_Multinf_R1_Den1_Freq4_Hradar.png')
