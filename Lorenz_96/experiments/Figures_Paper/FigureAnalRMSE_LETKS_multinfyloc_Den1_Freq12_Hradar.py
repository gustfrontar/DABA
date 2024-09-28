import matplotlib.pyplot as plt
import pickle
import numpy as np
import common_function as cf
plt.rcParams['text.usetex'] = True

plt.rcParams.update({'font.size': 22})

ObsErr='1'
Method='LETKS'
Ptemp='2.0'
Freq='4'
Den='1.0'
ObsOpe='3'


#NatureName='NatureR03_Den1_Freq12_Hlinear'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_' + Method + '_ptemp' + Ptemp + '_MultipleNature_Nature_Freq' + Freq + '_Den' + Den + '_Type' + ObsOpe + '_ObsErr' + ObsErr + '.npz'



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
    
for kk in range(len(temp_range)) :    
    total_analysis_rmse[:,:,kk] = cf.outlier_rmse_filter( total_analysis_rmse[:,:,kk] )

    min_error.append( str( np.round( np.nanmin( total_analysis_rmse[:,:,kk] ) , 2) ) )
    min_error_loc =  np.where(total_analysis_rmse[:,:,kk] == np.nanmin( total_analysis_rmse[:,:,kk] ) ) 
    x_error_loc[kk] = min_error_loc[0][0]
    y_error_loc[kk] = min_error_loc[1][0]

fig , axs = plt.subplots( 4 , 3 , figsize=(18,24) )

for kk in range(len(temp_range)) : 

   pcolor0=axs[0,kk].pcolor( mult_inf_range , loc_scale_range  , total_analysis_rmse[:,:,kk].T , vmin=0 , vmax=2 , cmap='YlGn' )
   axs[0,kk].plot(mult_inf_range[int(x_error_loc[kk])],loc_scale_range[int(y_error_loc[kk])],'ok')
   #plt.colorbar(pcolor0,ax=axs[0,0])
   axs[0,kk].set_ylabel('Loc. Scale')
   axs[0,kk].set_title('(a) - Min. RMSE=' + min_error[kk] )





plt.savefig('FigureAnalRMSE_'+Method+'_multinfyloc_Den'+Den+'_Freq'+Freq+'_ObsOpe' +ObsOpe +'.png')
