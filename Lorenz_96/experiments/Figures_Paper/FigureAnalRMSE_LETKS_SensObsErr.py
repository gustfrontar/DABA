import matplotlib.pyplot as plt
import pickle
import numpy as np
import common_function as cf
plt.rcParams['text.usetex'] = True

plt.rcParams.update({'font.size': 22})

Method='LETKF'
Ptemp='2.0'
Freq='20'
Den='0.5'
ObsOpe='1'
ObsErrList = ['0.3','1','5','25']

fig , axs = plt.subplots( 4 , 3 , figsize=(20,24)  )

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

    
    x_error_loc = np.zeros( len(temp_range) )
    y_error_loc = np.zeros( len(temp_range) )
    min_error = list()
    max_error = list()
    for kk in range(len(temp_range)) :    
        total_analysis_rmse[:,:,kk] = cf.outlier_rmse_filter( total_analysis_rmse[:,:,kk] )
        max_error.append( min( ( np.round( np.nanmax( total_analysis_rmse[:,:,kk] ) , 2) ) , 4) )
        min_error.append( ( np.round( np.nanmin( total_analysis_rmse[:,:,kk] ) , 2) ))
        min_error_loc =  np.where(total_analysis_rmse[:,:,kk] == np.nanmin( total_analysis_rmse[:,:,kk] ) ) 
        x_error_loc[kk] = min_error_loc[0][0]
        y_error_loc[kk] = min_error_loc[1][0]


    for kk in range(len(temp_range)) : 

        pcolor0=axs[iObsErr,kk].pcolor( mult_inf_range , loc_scale_range  , total_analysis_rmse[:,:,kk].T , vmin=min(min_error) , vmax=max(max_error) , cmap='YlGn' )
        axs[iObsErr,kk].plot(mult_inf_range[int(x_error_loc[kk])],loc_scale_range[int(y_error_loc[kk])],'ok')
        
        if kk == 0 :
           axs[iObsErr,kk].set_ylabel('Loc. Scale')
        if iObsErr == ( len( ObsErrList ) -1 ) :
           axs[iObsErr,kk].set_xlabel('Inflation')
        #if kk == ( len(temp_range) -1 ) :
        #    plt.colorbar(pcolor0,ax=axs[iObsErr,kk])
        axs[iObsErr,kk].set_title('Min. RMSE=' + str( min_error[kk] ) )
    fig.colorbar( pcolor0 , ax=axs[iObsErr,:] )




plt.savefig('FigureAnalRMSE_SensObsErr_'+Method+'_multinfyloc_Den'+Den+'_Freq'+Freq+'_ObsOpe' +ObsOpe +'.png')
