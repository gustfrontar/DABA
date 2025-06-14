import matplotlib.pyplot as plt
import pickle
import numpy as np
import common_function as cf
plt.rcParams['text.usetex'] = True

plt.rcParams.update({'font.size': 22})

NatureName='NatureR03_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF_' + NatureName + '.npz'



f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
NormalEnd=NormalEnd.astype(bool)  

        
total_analysis_rmse[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan      

total_analysis_rmse = cf.outlier_rmse_filter( total_analysis_rmse )

min_error = str( np.round( np.nanmin( total_analysis_rmse ) , 2) ) 
min_error_loc =  np.where(total_analysis_rmse == np.nanmin( total_analysis_rmse ) ) 
x_error_loc = min_error_loc[0][0]
y_error_loc = min_error_loc[1][0]

fig , axs = plt.subplots( 4 , 3 , figsize=(18,24) )

pcolor0=axs[0,0].pcolor( mult_inf_range , loc_scale_range  , total_analysis_rmse.T , vmin=0 , vmax=2.0 , cmap='YlGn' )
axs[0,0].plot(mult_inf_range[x_error_loc],loc_scale_range[y_error_loc],'ok')
#plt.colorbar(pcolor0,ax=axs[0,0])
axs[0,0].set_ylabel('Loc. Scale')

axs[0,0].set_title('(a) - Min. RMSE=' + min_error )


NatureName='NatureR1_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
NormalEnd=NormalEnd.astype(bool)    
        
total_analysis_rmse[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan  

total_analysis_rmse = cf.outlier_rmse_filter( total_analysis_rmse )
    
min_error = str( np.round( np.nanmin( total_analysis_rmse ) , 2) )  
min_error_loc =  np.where(total_analysis_rmse == np.nanmin( total_analysis_rmse ) ) 
x_error_loc = min_error_loc[0][0]
y_error_loc = min_error_loc[1][0]

pcolor0=axs[1,0].pcolor( mult_inf_range , loc_scale_range  , total_analysis_rmse.T , vmin=0 , vmax=2.0 , cmap='YlGn' )
axs[1,0].plot(mult_inf_range[x_error_loc],loc_scale_range[y_error_loc],'ok')
#plt.colorbar(pcolor0,ax=axs[1,0])
axs[1,0].set_ylabel('Loc. Scale')

axs[1,0].set_title('(b) - Min. RMSE=' + min_error )


NatureName='NatureR5_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
NormalEnd=NormalEnd.astype(bool)    
        
total_analysis_rmse[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan 

total_analysis_rmse = cf.outlier_rmse_filter( total_analysis_rmse )
     
min_error = str( np.round( np.nanmin( total_analysis_rmse ) , 2) )  
min_error_loc =  np.where(total_analysis_rmse == np.nanmin( total_analysis_rmse ) ) 
x_error_loc = min_error_loc[0][0]
y_error_loc = min_error_loc[1][0]

pcolor0=axs[2,0].pcolor( mult_inf_range , loc_scale_range  , total_analysis_rmse.T , vmin=0 , vmax=2.0 , cmap='YlGn' )
axs[2,0].plot(mult_inf_range[x_error_loc],loc_scale_range[y_error_loc],'ok')
#plt.colorbar(pcolor0,ax=axs[2,0])
axs[2,0].set_ylabel('Loc. Scale')

axs[2,0].set_title('(c) - Min. RMSE=' + min_error )


NatureName='NatureR25_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
NormalEnd=NormalEnd.astype(bool)    
        
total_analysis_rmse[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan   

total_analysis_rmse = cf.outlier_rmse_filter( total_analysis_rmse )
total_analysis_rmse_R25 = np.copy( total_analysis_rmse )
   
min_error = str( np.round( np.nanmin( total_analysis_rmse ) , 2) )  
min_error_loc =  np.where(total_analysis_rmse == np.nanmin( total_analysis_rmse ) ) 
x_error_loc = min_error_loc[0][0]
y_error_loc = min_error_loc[1][0]

pcolor0=axs[3,0].pcolor( mult_inf_range , loc_scale_range  , total_analysis_rmse.T , vmin=0 , vmax=2.0 , cmap='YlGn' )
#plt.colorbar(pcolor0,ax=axs[3,0])
axs[3,0].plot(mult_inf_range[x_error_loc],loc_scale_range[y_error_loc],'ok')
axs[3,0].set_ylabel('Loc. Scale')
axs[3,0].set_xlabel('Mult. Inf.')

axs[3,0].set_title('(d) - Min. RMSE=' + min_error )


#T2


NatureName='NatureR03_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF-T2_ptemp2.0_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
NormalEnd=NormalEnd.astype(bool)    
        
total_analysis_rmse[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan  

total_analysis_rmse = cf.outlier_rmse_filter( total_analysis_rmse )
    
min_error = str( np.round( np.nanmin( total_analysis_rmse ) , 2) )  
min_error_loc =  np.where(total_analysis_rmse == np.nanmin( total_analysis_rmse ) ) 
x_error_loc = min_error_loc[0][0]
y_error_loc = min_error_loc[1][0]

pcolor0=axs[0,1].pcolor( mult_inf_range , loc_scale_range  , total_analysis_rmse.T , vmin=0 , vmax=2.0 , cmap='YlGn' )
axs[0,1].plot(mult_inf_range[x_error_loc],loc_scale_range[y_error_loc],'ok')
#plt.colorbar(pcolor0,ax=axs[0,1])
#axs[0,1].set_ylabel('Mult. Inf.')
axs[0,1].set_title('(e) - Min. RMSE=' + min_error )


NatureName='NatureR1_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF-T2_ptemp2.0_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
NormalEnd=NormalEnd.astype(bool)    
        
total_analysis_rmse[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan   

total_analysis_rmse = cf.outlier_rmse_filter( total_analysis_rmse )

min_error = str( np.round( np.nanmin( total_analysis_rmse ) , 2) )  
min_error_loc =  np.where(total_analysis_rmse == np.nanmin( total_analysis_rmse ) ) 
x_error_loc = min_error_loc[0][0]
y_error_loc = min_error_loc[1][0]   

pcolor0=axs[1,1].pcolor( mult_inf_range , loc_scale_range  , total_analysis_rmse.T , vmin=0 , vmax=2.0 , cmap='YlGn' )
axs[1,1].plot(mult_inf_range[x_error_loc],loc_scale_range[y_error_loc],'ok')
#plt.colorbar(pcolor0,ax=axs[1,1])
#axs[1,1].set_ylabel('Mult. Inf.')

axs[1,1].set_title('(f) - Min. RMSE=' + min_error )


NatureName='NatureR5_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF-T2_ptemp2.0_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
NormalEnd=NormalEnd.astype(bool)    
        
total_analysis_rmse[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan   

total_analysis_rmse = cf.outlier_rmse_filter( total_analysis_rmse )
   
min_error = str( np.round( np.nanmin( total_analysis_rmse ) , 2) )  
min_error_loc =  np.where(total_analysis_rmse == np.nanmin( total_analysis_rmse ) ) 
x_error_loc = min_error_loc[0][0]
y_error_loc = min_error_loc[1][0]

pcolor0=axs[2,1].pcolor( mult_inf_range , loc_scale_range  , total_analysis_rmse.T , vmin=0 , vmax=2.0 , cmap='YlGn' )
axs[2,1].plot(mult_inf_range[x_error_loc],loc_scale_range[y_error_loc],'ok')
#plt.colorbar(pcolor0,ax=axs[2,1])
#axs[2,1].set_ylabel('Mult. Inf.')

axs[2,1].set_title('(g) - Min. RMSE=' + min_error )


NatureName='NatureR25_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF-T2_ptemp2.0_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
NormalEnd=NormalEnd.astype(bool)    
        
total_analysis_rmse[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan 

total_analysis_rmse = cf.outlier_rmse_filter( total_analysis_rmse )
#total_analysis_rmse[:,-1]=np.nan

min_error = str( np.round( np.nanmin( total_analysis_rmse ) , 2) )  
min_error_loc =  np.where(total_analysis_rmse == np.nanmin( total_analysis_rmse ) ) 
x_error_loc = min_error_loc[0][0]
y_error_loc = min_error_loc[1][0]
 
#diff_rmse = 100*( ( total_analysis_rmse - total_analysis_rmse_R25 ) / total_analysis_rmse_R25 ).T
     
pcolor0=axs[3,1].pcolor( mult_inf_range , loc_scale_range  , total_analysis_rmse.T , vmin=0 , vmax=2.0 , cmap='YlGn' )
axs[3,1].plot(mult_inf_range[x_error_loc],loc_scale_range[y_error_loc],'ok')
#plt.colorbar(pcolor0,ax=axs[3,1])
axs[3,1].set_xlabel('Mult. Inf.')

axs[3,1].set_title('(h) - Min. RMSE=' + min_error )

#T3


NatureName='NatureR03_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF-T3_ptemp2.0_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
NormalEnd=NormalEnd.astype(bool)    
        
total_analysis_rmse[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan  

total_analysis_rmse = cf.outlier_rmse_filter( total_analysis_rmse )
    
min_error = str( np.round( np.nanmin( total_analysis_rmse ) , 2) )  
min_error_loc =  np.where(total_analysis_rmse == np.nanmin( total_analysis_rmse ) ) 
x_error_loc = min_error_loc[0][0]
y_error_loc = min_error_loc[1][0]

pcolor0=axs[0,2].pcolor( mult_inf_range , loc_scale_range  , total_analysis_rmse.T , vmin=0 , vmax=2.0 , cmap='YlGn' )
axs[0,2].plot(mult_inf_range[x_error_loc],loc_scale_range[y_error_loc],'ok')
#plt.colorbar(pcolor0,ax=axs[0,1])
#axs[0,1].set_ylabel('Mult. Inf.')
axs[0,2].set_title('(e) - Min. RMSE=' + min_error )


NatureName='NatureR1_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF-T3_ptemp2.0_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
NormalEnd=NormalEnd.astype(bool)    
        
total_analysis_rmse[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan   

total_analysis_rmse = cf.outlier_rmse_filter( total_analysis_rmse )

min_error = str( np.round( np.nanmin( total_analysis_rmse ) , 2) )  
min_error_loc =  np.where(total_analysis_rmse == np.nanmin( total_analysis_rmse ) ) 
x_error_loc = min_error_loc[0][0]
y_error_loc = min_error_loc[1][0]   

pcolor0=axs[1,2].pcolor( mult_inf_range , loc_scale_range  , total_analysis_rmse.T , vmin=0 , vmax=2.0 , cmap='YlGn' )
axs[1,2].plot(mult_inf_range[x_error_loc],loc_scale_range[y_error_loc],'ok')
#plt.colorbar(pcolor0,ax=axs[1,1])
#axs[1,1].set_ylabel('Mult. Inf.')

axs[1,2].set_title('(f) - Min. RMSE=' + min_error )


NatureName='NatureR5_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF-T3_ptemp2.0_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
NormalEnd=NormalEnd.astype(bool)    
        
total_analysis_rmse[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan   

total_analysis_rmse = cf.outlier_rmse_filter( total_analysis_rmse )
   
min_error = str( np.round( np.nanmin( total_analysis_rmse ) , 2) )  
min_error_loc =  np.where(total_analysis_rmse == np.nanmin( total_analysis_rmse ) ) 
x_error_loc = min_error_loc[0][0]
y_error_loc = min_error_loc[1][0]

pcolor0=axs[2,2].pcolor( mult_inf_range , loc_scale_range  , total_analysis_rmse.T , vmin=0 , vmax=2.0 , cmap='YlGn' )
axs[2,2].plot(mult_inf_range[x_error_loc],loc_scale_range[y_error_loc],'ok')
#plt.colorbar(pcolor0,ax=axs[2,1])
#axs[2,1].set_ylabel('Mult. Inf.')

axs[2,2].set_title('(g) - Min. RMSE=' + min_error )


NatureName='NatureR25_Den1_Freq4_Hradar'
exp_filename='../npz/Sesitivity_experiment_multinfyloc_LETKF-T3_ptemp2.0_' + NatureName + '.npz'

f=open(exp_filename,'rb')
[results,mult_inf_range,loc_scale_range,Alpha_temp_list,total_analysis_rmse,total_forecast_rmse,total_analysis_sprd,total_forecast_sprd] = pickle.load(f)
f.close()

NormalEnd = np.zeros( (len(mult_inf_range) , len(loc_scale_range) ))
for ii in range(len(mult_inf_range)) :
    for jj in range(len(loc_scale_range)) :
        NormalEnd[ii,jj] = 1-results[ii*len(loc_scale_range)+jj]['NormalEnd']
NormalEnd=NormalEnd.astype(bool)    
        
total_analysis_rmse[NormalEnd] = np.nan        
total_analysis_sprd[NormalEnd] = np.nan 

total_analysis_rmse = cf.outlier_rmse_filter( total_analysis_rmse )
#total_analysis_rmse[:,-1]=np.nan

min_error = str( np.round( np.nanmin( total_analysis_rmse ) , 2) )  
min_error_loc =  np.where(total_analysis_rmse == np.nanmin( total_analysis_rmse ) ) 
x_error_loc = min_error_loc[0][0]
y_error_loc = min_error_loc[1][0]
 
#diff_rmse = 100*( ( total_analysis_rmse - total_analysis_rmse_R25 ) / total_analysis_rmse_R25 ).T
     
pcolor0=axs[3,2].pcolor( mult_inf_range , loc_scale_range  , total_analysis_rmse.T , vmin=0 , vmax=2.0 , cmap='YlGn' )
axs[3,2].plot(mult_inf_range[x_error_loc],loc_scale_range[y_error_loc],'ok')
#plt.colorbar(pcolor0,ax=axs[3,1])
axs[3,2].set_xlabel('Mult. Inf.')

axs[3,2].set_title('(h) - Min. RMSE=' + min_error )


fig.subplots_adjust(right=0.84)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
fig.colorbar(pcolor0,cax=cbar_ax)


plt.savefig('FigureAnalRMSE_multinfyloc_Den1_Freq4_Hradar.png')
