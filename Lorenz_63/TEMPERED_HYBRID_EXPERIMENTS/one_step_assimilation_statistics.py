#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:12:12 2020

@author: jruiz
"""
#%%
#!/usr/bin/env python
# coding: utf-8
#En este experimento genero el prior y el posterior usando un ensamble de muchos miembros.
#Luego tomo un subsample pequenio y aplico diferentes estrategias para ver como aproximan al posterior.



import sys
from plots import plot_xyz_evol 
from aux_functions import *
import time
from scipy import stats

sys.path.append("../")

import os
os.environ["OMP_NUM_THREADS"]="20"



#Importamos todos los modulos que van a ser usados en esta notebook
import numpy as np
import scipy as sc
from lorenz_63  import lorenz63 as model        #Import the model (fortran routines)
import Lorenz_63_DA as da
import matplotlib.pyplot as plt
import pickle as pkl

from lorenz_63  import lorenz63 as model        #Import the model (fortran routines)

#Seleccionar aqui el operador de las observaciones que se desea usar.
from Lorenz_63_ObsOperator import forward_operator_logaritmic        as forward_operator
from Lorenz_63_ObsOperator import forward_operator_logaritmic_ens    as forward_operator_ens
from Lorenz_63_ObsOperator import forward_operator_logaritmic_tl     as forward_operator_tl

#Obtengo el numero de observaciones (lo obtengo directamente del forward operator)
nobs=np.size(forward_operator(np.array([0,0,0])))

#------------------------------------------------------------
# Especificamos los parametros que usara el modelo
#------------------------------------------------------------

a      = 10.0      # standard L63 10.0 
r      = 28.0      # standard L63 28.0
b      = 8.0/3.0   # standard L63 8.0/3.0
p=np.array([a,r,b])
dt=0.01            # Paso de tiempo para la integracion del modelo de Lorenz

numtrans=1000
bst=32
EnsSize=10
nvars=3



R0=0.05
R=R0*np.eye(nobs)
np.random.seed(20)  #Fix the random seed.


f=open('Ens_'+str(bst)+ '_' + str(numtrans) + '.pkl','rb')
[xa0_ens,xf1_ens] = pkl.load(f)
xa0_ens=xa0_ens[:,0:10000]
xf1_ens=xf1_ens[:,0:10000]
f.close()  

SampleSize=xa0_ens.shape[1]

NObs= forward_operator( np.array([0,0,0]) ).size

#First chose a random member which will be the true. We can chose the last one insted of a random one.

method_list=['ETKF_ONE_STEP']
NIter=5                          #Number of iterations for RIP and TEMP


NRep=1   #Number of repetition of each assimilation experiment.
xt1 = np.zeros((nvars,NRep))
yo1 = np.zeros((NObs,NRep))
xf1_ens_da= np.zeros((nvars,EnsSize,NRep))
xa0_ens_da= np.zeros((nvars,EnsSize,NRep))


metrics=dict()
analysis=dict()

#------------------------------------------------------------------------------
#  Compute metrics
#------------------------------------------------------------------------------

for irep in range( NRep )   :
    
    start_iteration = time.time()

    #First choose N ens + 1 elements from the large sample. The first will be the true 
    #state of the system. The following ones will be the ensemble members
    
    tmp_index = np.random.choice(SampleSize, size=(EnsSize+1), replace=False )
    
    #We get the true
    xt1[:,irep] = xf1_ens[:,tmp_index[0]]    
    #We get the forecast ensemble
    xf1_ens_da[:,:,irep] = xf1_ens[:,tmp_index[1:]] 
    xa0_ens_da[:,:,irep] = xa0_ens[:,tmp_index[1:]] 
    #We get the observation randomly perturbing the true state
    yo1[:,irep] = forward_operator( xt1[:,irep] ) + np.random.multivariate_normal(np.zeros(nobs),R*np.eye(nobs))
    
    
    #Third compute the full posterior density using importance sampling
    w=np.zeros(xa0_ens.shape[1])
    yf1_ens = forward_operator_ens( xf1_ens )
    
    #Compute the weights of each member of the sample
    dep = np.tile(yo1[:,irep],(SampleSize,1)).T - yf1_ens[:]
    if nobs == 1 :
       w=  np.exp( -np.power( dep , 2 )  / (2.0*R0) ) 
    else         :
       w=  np.exp( np.sum( -np.power( dep , 2 ) , 0 )  / (2.0*R0) )     
    w = w / np.sum(w) 
    
    #Compute the kernel for the full sample to describe the prior and the posterior.
    kernel_xf1    = stats.gaussian_kde(xf1_ens)
    kernel_xa1    = stats.gaussian_kde(xf1_ens,weights=w)
    #Compute the kernel representation of the prior as seen by the ensemble.
    kernel_xf1_da = stats.gaussian_kde(xf1_ens_da[:,:,irep])
    
    #Compute the true covariance matrix for the prior and the posterior
    [xf1_mean , xf1_cov ]=mean_covar( xf1_ens )
    [xa1_mean , xa1_cov ]=mean_covar( xf1_ens , w=w)
    #Compute the covariance matrix for the ensemble representation of the prior.
    [xf1_mean_da , xf1_cov_da ]=mean_covar( xf1_ens )
    
    for my_method in method_list   :
    
       if my_method == 'ETKF_ONE_STEP'    :
           state_ens = xf1_ens_da[:,:,irep]
           [ state_ens , null_var , null_var , null_var , null_var ] =da.analysis_update_ETKF( yo1[:,irep] , state_ens , forward_operator , R , 1.0 )
       
       
       if irep == 0:
           analysis[my_method] = np.zeros((nvars,NEns,NRep))
           metrics[my_method] = dict()
           metrics[my_method]['dist_covar_a']=np.zeros(NRep)
           metrics[my_method]['dist_covar_f']=np.zeros(NRep)
           metrics[my_method]['dist_var_a']=np.zeros(NRep)
           metrics[my_method]['dist_var_f']=np.zeros(NRep)
           metrics[my_method]['bias_var_a']=np.zeros(NRep)
           metrics[my_method]['bias_var_f']=np.zeros(NRep)
           metrics[my_method]['kld_a']=np.zeros(NRep)
           metrics[my_method]['kld_f']=np.zeros(NRep)
           metrics[my_method]['bias_f']=np.zeros(NRep)
           metrics[my_method]['bias_a']=np.zeros(NRep)
           metrics[my_method]['rmse_f']=np.zeros(NRep)
           metrics[my_method]['rmse_a']=np.zeros(NRep)
    
       #Save the analysis ensemble   
       analysis[my_method][:,:,irep] = state_ens 
       #Compute the mean and covariance matrix of the analysis ensemble
       [xa1_mean_da , xa1_cov_da ]=mean_covar( state_ens )
       #Compute the kernel representation of the prior as seen by the ensemble.
       kernel_xf1_da = stats.gaussian_kde(xf1_ens_da[:,:,irep])
       #Compute different metrics to compare the ensemble based posterior and the real posterior.
       metrics['dist_covar_a'][irep]=dist_covar( xa1_cov_da , xa1_cov )
       metrics['dist_covar_f'][irep]=dist_covar( xf1_cov_da , xf1_cov )
       metrics['dist_var_a'][irep]=dist_var( xa1_cov_da , xa1_cov )
       metrics['dist_var_f'][irep]=dist_var( xf1_cov_da , xf1_cov )
       metrics['bias_var_a'][irep]=bias_var( xa1_cov_da , xa1_cov )
       metrics['bias_var_f'][irep]=bias_var( xf1_cov_da , xf1_cov )
       metrics['kld_a'][irep]=kld3d_kde(  kernel_xf1 , kernel_xf1_da  )
       metrics['kld_f'][irep]=kld3d_kde(  kernel_xa1 , kernel_xa1_da  )
       metrics['bias_f'][irep]=bias_mean( xf1_mean_da , xf1_mean )
       metrics['bias_a'][irep]=bias_mean( xa1_mean_da , xa1_mean )
       metrics['rmse_f'][irep]=rmse_mean( xf1_mean_da , xf1_mean )
       metrics['rmse_a'][irep]=rmse_mean( xa1_mean_da , xa1_mean )
       
       print('Iteration took ', time.time()-start_iteration, 'seconds.') 



kernel_xf1    = stats.gaussian_kde(xf1_ens)
X, Y = np.mgrid[-20:20:100j, -20:20:100j]
positions = np.vstack([X.ravel(), Y.ravel() , 40*np.ones(100*100)])
Z = np.reshape(kernel_xf1(p
                          ositions).T, X.shape)

import matplotlib.pyplot as plt
plt.pcolor(X,Y,Z)

plt.xlim([-20,0])
plt.ylim([-20,0])
plt.show()

