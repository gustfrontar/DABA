import matplotlib.pyplot as plt
import pickle
import numpy as np
import common_function as cf
plt.rcParams['text.usetex'] = True

plt.rcParams.update({'font.size': 22})

ObsErr='1'
Method='LETKS'
Ptemp='2.0'
Freq='12'
Den='1.0'
ObsOpe='1'
ObsErrList = ['0.3','1','5','25']

fig , axs = plt.subplots( 4 , 3 , figsize=(18,24) )

for iObsErr , ObsErr in enumerate( ObsErrList ) :
#NatureName='NatureR03_Den1_Freq12_Hlinear'
    exp_filename='../npz/Sensitivity_experiment_multinfyloc_' + Method + '_ptemp' + Ptemp + '_MultipleNature_Nature_Freq' + Freq + '_Den' + Den + '_Type' + ObsOpe + '_ObsErr' + ObsErr + '.npz'



    f=open(exp_filename,'rb')
    Output = pickle.load(f)
    f.close()
    mult_inf_range = Output['MultInfRange']
    loc_scale_range = Output['LocScaleRange']
    temp_range = Output['NTempRange']
    total_analysis_rmse=Output['total_analysis_rmse']

    NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) , len(temp_range) ))
    for ii in range(len(mult_inf_range)) :
        for jj in range(len(loc_scale_range)) :
            for kk in range( len(temp_range))  :
                NormalEnd[ii,jj,kk] = ~np.any( np.isnan( Output['XAMean'][ii,jj,kk,:,:] ) ) 

    
    x_error_loc = np.zeros( len(temp_range) ).astype(int)
    y_error_loc = np.zeros( len(temp_range) ).astype(int)
    min_error = list()
    
    for kk in range(len(temp_range)) :    
        total_analysis_rmse[:,:,kk] = cf.outlier_rmse_filter( total_analysis_rmse[:,:,kk] )

        min_error.append( str( np.round( np.nanmin( total_analysis_rmse[:,:,kk] ) , 2) ) )
        min_error_loc =  np.where(total_analysis_rmse[:,:,kk] == np.nanmin( total_analysis_rmse[:,:,kk] ) ) 
        x_error_loc[kk] =  min_error_loc[0][0] 
        y_error_loc[kk] =  min_error_loc[1][0] 


    for kk in range(len(temp_range)) : 

        pcolor0=axs[iObsErr,kk].pcolor( Output['XAMean'][x_error_loc[kk],y_error_loc[kk],kk,0:20,-100:] - Output['XNature'][0:20,-100:] , vmin=-3 , vmax=3 , cmap='bwr' )
        #axs[iObsErr,kk].plot(mult_inf_range[int(x_error_loc[kk])],loc_scale_range[int(y_error_loc[kk])],'ok')
        #plt.colorbar(pcolor0,ax=axs[0,0])
        axs[iObsErr,kk].set_ylabel('X')
        axs[iObsErr,kk].set_xlabel('Time')
        axs[iObsErr,kk].set_title('Min. RMSE=' + min_error[kk] )





plt.savefig('Figure_AnalysisError_'+Method+'_multinfyloc_Den'+Den+'_Freq'+Freq+'_ObsOpe' +ObsOpe +'.png')
