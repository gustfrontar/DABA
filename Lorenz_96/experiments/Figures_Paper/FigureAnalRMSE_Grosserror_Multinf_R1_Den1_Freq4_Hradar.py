import matplotlib.pyplot as plt
import pickle
import numpy as np
plt.rcParams['text.usetex'] = True

plt.rcParams.update({'font.size': 22})

NatureName='NatureR1_Den1_Freq4_Hradar'

exp_filename='../npz/Sesitivity_experiment_tempering_grosserr_LETKF_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,gross_error_check_range, min_dbz_thresh_range,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(gross_error_check_range) , len(min_dbz_thresh_range) ))
for ii in range(len(gross_error_check_range)) :
    for jj in range(len(min_dbz_thresh_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(min_dbz_thresh_range)+jj]['NormalEnd']
NormalEnd=NormalEnd.astype(bool)    
        
total_analysis_rmse[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan      

min_dbz_thresh_range = 1.0 - min_dbz_thresh_range

fig , axs = plt.subplots( 2 , 2 , figsize=(11,12) )

pcolor0=axs[0,0].pcolor( min_dbz_thresh_range, gross_error_check_range , total_analysis_rmse , vmin=0 , vmax=5.0 , cmap='YlGn' )
plt.colorbar(pcolor0,ax=axs[0,0])
axs[0,0].set_ylabel(r'$\beta$')

axs[0,0].set_title('(a)')

pcolor1=axs[0,1].pcolor( min_dbz_thresh_range, gross_error_check_range , total_analysis_sprd , vmin=0 , vmax=5.0 , cmap='YlGn' )
plt.colorbar(pcolor1,ax=axs[0,1])
axs[0,1].set_title('(b)')

min_error = np.round( np.min( total_analysis_rmse , axis = 0 ) , 2 )

NatureName='NatureR5_Den1_Freq4_Hradar'

exp_filename='../npz/Sesitivity_experiment_tempering_grosserr_LETKF_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,gross_error_check_range, min_dbz_thresh_range,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(gross_error_check_range) , len(min_dbz_thresh_range) ))
for ii in range(len(gross_error_check_range)) :
    for jj in range(len(min_dbz_thresh_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(min_dbz_thresh_range)+jj]['NormalEnd']
NormalEnd=NormalEnd.astype(bool) 

total_analysis_rmse[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan   

min_dbz_thresh_range = 1.0 - min_dbz_thresh_range

pcolor2=axs[1,0].pcolor( min_dbz_thresh_range, gross_error_check_range , total_analysis_rmse , vmin=0 , vmax=5.0 , cmap='YlGn' )
plt.colorbar(pcolor2,ax=axs[1,0])
axs[1,0].set_xlabel(r'$T_{nr}$')
axs[1,0].set_ylabel(r'$\beta$')
axs[1,0].set_title('(c)')

pcolor3=axs[1,1].pcolor( min_dbz_thresh_range, gross_error_check_range , total_analysis_sprd , vmin=0 , vmax=5.0 , cmap='YlGn' )
plt.colorbar(pcolor3,ax=axs[1,1])
axs[1,1].set_xlabel(r'$T_{nr}$')
axs[1,1].set_title('(d)')


plt.savefig('FigureAnalRMSE_Grosserror_Multinf_R1_Den1_Freq4_Hradar.png')
