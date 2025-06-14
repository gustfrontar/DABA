import matplotlib.pyplot as plt
import pickle
import numpy as np
import common_function as cf
plt.rcParams['text.usetex'] = True

plt.rcParams.update({'font.size': 22})

ObsErr='1'
Method='LETKF'
Ptemp='2.0'
Freq='4'
Den='1.0'
ObsOpe='3'

exp_filename='../npz/Sensitivity_experiment_multinfyloc_' + Method + '_ptemp' + Ptemp + '_MultipleNature_Nature_Freq' + Freq + '_Den' + Den + '_Type' + ObsOpe + '_ObsErr' + ObsErr + '.npz'

f=open(exp_filename,'rb')
Output = pickle.load(f)
f.close()

Nature = Output['XNature'][:,:]
XAMean = Output['XAMean'][:,:,1,:,:]
XFMean = Output['XFMean'][:,:,1,:,:]
total_analysis_rmse = Output['total_analysis_rmse'][:,:,:]

total_analysis_rmse_smoothed = total_analysis_rmse.copy()

for ii in range(1,np.shape(total_analysis_rmse)[0]-1):
    for jj in range(1,np.shape(total_analysis_rmse)[1]-1):
        total_analysis_rmse_smoothed[ii,jj,:] = total_analysis_rmse[ii,jj,:] + total_analysis_rmse[ii,jj-1,:] + total_analysis_rmse[ii-1,jj,:] + total_analysis_rmse[ii,jj+1,:] + total_analysis_rmse[ii+1,jj,:]
        total_analysis_rmse_smoothed[ii,jj,:] = total_analysis_rmse_smoothed[ii,jj,:] / 5.

min_index_letkf = np.unravel_index(np.argmin(total_analysis_rmse_smoothed[:,:,0], axis=None), total_analysis_rmse_smoothed[:,:,0].shape)

min_index_tletkf = np.unravel_index(np.argmin(total_analysis_rmse_smoothed[:,:,1], axis=None), total_analysis_rmse_smoothed[:,:,1].shape)
        
print(min_index_letkf)
print(min_index_tletkf)
