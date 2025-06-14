import matplotlib.pyplot as plt
import pickle
import numpy as np
import common_function as cf
plt.rcParams['text.usetex'] = True

plt.rcParams.update({'font.size': 22})

NatureName='NatureR03_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF_F7.5_' + NatureName + '.npz'

fig , axs = plt.subplots( 4 , 2 , figsize=(18,24) )

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
analysis_bias = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
        analysis_bias[ii,jj] = np.mean( results[ii*total_analysis_rmse.shape[1]+jj]['XATBias'][200:] ) 

NormalEnd=NormalEnd.astype(bool)  

        
analysis_bias[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan      

#analysis_bias = cf.outlier_rmse_filter( analysis_bias )

min_error = str( np.round( np.nanmin( np.abs(analysis_bias) ) , 2) ) 
min_error_loc =  np.where( np.abs( analysis_bias ) == np.nanmin( np.abs( analysis_bias ) ) ) 
x_error_loc = min_error_loc[0][0]
y_error_loc = min_error_loc[1][0]



pcolor0=axs[0,0].pcolor( mult_inf_range , loc_scale_range  , analysis_bias.T , vmin=-0.25 , vmax=0.25 , cmap='bwr' )
axs[0,0].plot(mult_inf_range[x_error_loc],loc_scale_range[y_error_loc],'ok')

axs[0,0].set_ylabel('Loc. Scale')

axs[0,0].set_title('(a) - Min. BIAS=' + min_error )


NatureName='NatureR1_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF_F7.5_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NNormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
analysis_bias = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
        analysis_bias[ii,jj] = np.mean( results[ii*total_analysis_rmse.shape[1]+jj]['XATBias'][200:] )

NormalEnd=NormalEnd.astype(bool)  

        
analysis_bias[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan      

#analysis_bias = cf.outlier_rmse_filter( analysis_bias )

min_error = str( np.round( np.nanmin( np.abs(analysis_bias) ) , 2) ) 
min_error_loc =  np.where( np.abs( analysis_bias ) == np.nanmin( np.abs( analysis_bias ) ) ) 
x_error_loc = min_error_loc[0][0]
y_error_loc = min_error_loc[1][0]

pcolor0=axs[1,0].pcolor( mult_inf_range , loc_scale_range  , analysis_bias.T , vmin=-0.25 , vmax=0.25 , cmap='bwr' )
axs[1,0].plot(mult_inf_range[x_error_loc],loc_scale_range[y_error_loc],'ok')

axs[1,0].set_ylabel('Loc. Scale')

axs[1,0].set_title('(b) - Min. BIAS=' + min_error )


NatureName='NatureR5_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF_F7.5_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NNormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
analysis_bias = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
        analysis_bias[ii,jj] = np.mean( results[ii*total_analysis_rmse.shape[1]+jj]['XATBias'][200:] )

NormalEnd=NormalEnd.astype(bool)  

        
analysis_bias[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan      

#analysis_bias = cf.outlier_rmse_filter( analysis_bias )

min_error = str( np.round( np.nanmin( np.abs(analysis_bias) ) , 2) ) 
min_error_loc =  np.where( np.abs( analysis_bias ) == np.nanmin( np.abs( analysis_bias ) ) ) 
x_error_loc = min_error_loc[0][0]
y_error_loc = min_error_loc[1][0]

pcolor0=axs[2,0].pcolor( mult_inf_range , loc_scale_range  , analysis_bias.T , vmin=-0.25 , vmax=0.25 , cmap='bwr' )
axs[2,0].plot(mult_inf_range[x_error_loc],loc_scale_range[y_error_loc],'ok')
#plt.colorbar(pcolor0,ax=axs[2,0])
axs[2,0].set_ylabel('Loc. Scale')

axs[2,0].set_title('(c) - Min. BIAS=' + min_error )


NatureName='NatureR25_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF_F7.5_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
analysis_bias = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
        analysis_bias[ii,jj] = np.mean( results[ii*total_analysis_rmse.shape[1]+jj]['XATBias'][200:] ) 

NormalEnd=NormalEnd.astype(bool)  

        
analysis_bias[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan      

#analysis_bias = cf.outlier_rmse_filter( analysis_bias )

min_error = str( np.round( np.nanmin( np.abs(analysis_bias) ) , 2) ) 
min_error_loc =  np.where( np.abs( analysis_bias ) == np.nanmin( np.abs( analysis_bias ) ) ) 
x_error_loc = min_error_loc[0][0]
y_error_loc = min_error_loc[1][0]



pcolor0=axs[3,0].pcolor( mult_inf_range , loc_scale_range  , analysis_bias.T , vmin=-0.25 , vmax=0.25 , cmap='bwr' )
axs[3,0].plot(mult_inf_range[x_error_loc],loc_scale_range[y_error_loc],'ok')

axs[3,0].set_ylabel('Loc. Scale')
axs[3,0].set_xlabel('Mult. Inf.')

axs[3,0].set_title('(d) - Min. BIAS=' + min_error )


#T2


NatureName='NatureR03_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF-T2_ptemp2.0_F7.5_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
analysis_bias = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
        analysis_bias[ii,jj] = np.mean( results[ii*total_analysis_rmse.shape[1]+jj]['XATBias'][200:] ) 

NormalEnd=NormalEnd.astype(bool)  

        
analysis_bias[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan      

#analysis_bias = cf.outlier_rmse_filter( analysis_bias )

min_error = str( np.round( np.nanmin( np.abs(analysis_bias) ) , 2) ) 
min_error_loc =  np.where( np.abs( analysis_bias ) == np.nanmin( np.abs( analysis_bias ) ) ) 
x_error_loc = min_error_loc[0][0]
y_error_loc = min_error_loc[1][0]



pcolor0=axs[0,1].pcolor( mult_inf_range , loc_scale_range  , analysis_bias.T , vmin=-0.25 , vmax=0.25 , cmap='bwr' )
axs[0,1].plot(mult_inf_range[x_error_loc],loc_scale_range[y_error_loc],'ok')

#plt.colorbar(pcolor0,ax=axs[0,1])
#axs[0,1].set_ylabel('Mult. Inf.')
axs[0,1].set_title('(e) - Min. BIAS=' + min_error )


NatureName='NatureR1_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF-T2_ptemp2.0_F7.5_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
analysis_bias = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
        analysis_bias[ii,jj] = np.mean( results[ii*total_analysis_rmse.shape[1]+jj]['XATBias'][200:] ) 

NormalEnd=NormalEnd.astype(bool)  

        
analysis_bias[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan      

#analysis_bias = cf.outlier_rmse_filter( analysis_bias )

min_error = str( np.round( np.nanmin( np.abs(analysis_bias) ) , 2) ) 
min_error_loc =  np.where( np.abs( analysis_bias ) == np.nanmin( np.abs( analysis_bias ) ) ) 
x_error_loc = min_error_loc[0][0]
y_error_loc = min_error_loc[1][0]



pcolor0=axs[1,1].pcolor( mult_inf_range , loc_scale_range  , analysis_bias.T , vmin=-0.25 , vmax=0.25 , cmap='bwr' )
axs[1,1].plot(mult_inf_range[x_error_loc],loc_scale_range[y_error_loc],'ok')

#plt.colorbar(pcolor0,ax=axs[1,1])
#axs[1,1].set_ylabel('Mult. Inf.')

axs[1,1].set_title('(f) - Min. BIAS=' + min_error )


NatureName='NatureR5_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF-T2_ptemp2.0_F7.5_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
analysis_bias = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
        analysis_bias[ii,jj] = np.mean( results[ii*total_analysis_rmse.shape[1]+jj]['XATBias'][200:] ) 

NormalEnd=NormalEnd.astype(bool)  

        
analysis_bias[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan      

#analysis_bias = cf.outlier_rmse_filter( analysis_bias )

min_error = str( np.round( np.nanmin( np.abs(analysis_bias) ) , 2) ) 
min_error_loc =  np.where( np.abs( analysis_bias ) == np.nanmin( np.abs( analysis_bias ) ) ) 
x_error_loc = min_error_loc[0][0]
y_error_loc = min_error_loc[1][0]



pcolor0=axs[2,1].pcolor( mult_inf_range , loc_scale_range  , analysis_bias.T , vmin=-0.25 , vmax=0.25 , cmap='bwr' )
axs[2,1].plot(mult_inf_range[x_error_loc],loc_scale_range[y_error_loc],'ok')

#plt.colorbar(pcolor0,ax=axs[2,1])
#axs[2,1].set_ylabel('Mult. Inf.')

axs[2,1].set_title('(g) - Min. BIAS=' + min_error )


NatureName='NatureR25_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF-T2_ptemp2.0_F7.5_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
analysis_bias = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
        analysis_bias[ii,jj] = np.mean( results[ii*total_analysis_rmse.shape[1]+jj]['XATBias'][200:] ) 

NormalEnd=NormalEnd.astype(bool)  

        
analysis_bias[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan      

#analysis_bias = cf.outlier_rmse_filter( analysis_bias )

min_error = str( np.round( np.nanmin( np.abs(analysis_bias) ) , 2) ) 
min_error_loc =  np.where( np.abs( analysis_bias ) == np.nanmin( np.abs( analysis_bias ) ) ) 
x_error_loc = min_error_loc[0][0]
y_error_loc = min_error_loc[1][0]



pcolor0=axs[3,1].pcolor( mult_inf_range , loc_scale_range  , analysis_bias.T , vmin=-0.25 , vmax=0.25 , cmap='bwr' )
axs[3,1].plot(mult_inf_range[x_error_loc],loc_scale_range[y_error_loc],'ok')

#plt.colorbar(pcolor0,ax=axs[3,1])
axs[3,1].set_xlabel('Mult. Inf.')

axs[3,1].set_title('(h) - Min. BIAS=' + min_error )


fig.subplots_adjust(right=0.84)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
fig.colorbar(pcolor0,cax=cbar_ax)


plt.savefig('FigureAnalBIAS_multinfyloc_F7.5_Den1_Freq4_Hradar.png')
