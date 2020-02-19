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


#TODO generar una figura similar a la de la evolucion pero en el espacio de las observaciones.


import sys
sys.path.append("../")

sys.path.append("./")
from plots import *
from aux_functions import *



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
from Lorenz_63_ObsOperator import forward_operator_nonlinearsum        as forward_operator
from Lorenz_63_ObsOperator import forward_operator_nonlinearsum_ens    as forward_operator_ens
from Lorenz_63_ObsOperator import forward_operator_nonlinearsum_tl     as forward_operator_tl

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
EnsSize=30
nvars=3

NTemp=5
NRip=5

R0=0.05
R=R0*np.eye(nobs)
np.random.seed(20)  #Fix the random seed.


f=open('Ens_'+str(bst)+ '_' + str(numtrans) + '.pkl','rb')
[xa0_ens,xf1_ens] = pkl.load(f)
f.close()  
#First chose a random member which will be the true. We can chose the last one insted of a random one.

xt0 = xa0_ens[:,-1]
xt1 = xf1_ens[:,-1]    #No model error

#Second generate an observation from the true
yo1 = forward_operator( xt1 ) + np.random.multivariate_normal(np.zeros(nobs),R*np.eye(nobs))
 
#Third compute the full posterior density using importance sampling
w=np.zeros(xa0_ens.shape[1])
yf1_ens = forward_operator_ens( xf1_ens )
dep = yo1 - yf1_ens[:]
w=  np.exp( -np.power( dep , 2 )  / (2.0*R0) )  #This is a simplified version of the computation of the likelihood.
w = w / np.sum(w) #Weigth normalization.

[xa0_mean , xa0_cov ]=mean_covar( xa0_ens ) 
[xf1_mean , xf1_cov ]=mean_covar( xf1_ens )
[xa1_mean , xa1_cov ]=mean_covar( xf1_ens , w=w)

#Third generate an ensemble of states that will represent the ensemble available for data assimilation.
#Podemos elegirlas de manera aleatoria o ir por las primeras EnsSize y ya... (voy por Homero Simpson y ya)

xa0_ens_da = xa0_ens[:,0:EnsSize]
xf1_ens_da = xf1_ens[:,0:EnsSize]
#Ahora vamos a definir diferentes variantes de asimilacion y vamos a guardar la evolucion
#entre la forma en la que aproximan el prior y la forma en la que aproximan el posterior y 
#todos los pasos intermedios si los ubiera.
#------------------------------------------------------------------------------
#  ETKF de un solo paso
#------------------------------------------------------------------------------

#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol=np.zeros((nvars,EnsSize,2)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.

tmp_for = np.copy( xf1_ens_da )
[ tmp_anal , tmp_anal_mean , Pa , null_var , null_var ] =da.analysis_update_ETKF( yo1 , tmp_for , forward_operator , R , 1.0 )

x_ens_evol[:,:,1] = tmp_anal

#Calculamos las densidades en un rango rodeando al true.
xmin = np.min([np.min(xt1[0]),np.min(x_ens_evol[0,:,:]) ]) - 2.0
xmax = np.max([np.max(xt1[0]),np.max(x_ens_evol[0,:,:]) ]) + 2.0
ymin = np.min([np.min(xt1[1]),np.min(x_ens_evol[1,:,:]) ]) - 2.0
ymax = np.max([np.max(xt1[1]),np.max(x_ens_evol[1,:,:]) ]) + 2.0
zmin = np.min([np.min(xt1[2]),np.min(x_ens_evol[2,:,:]) ]) - 2.0
zmax = np.max([np.max(xt1[2]),np.max(x_ens_evol[2,:,:]) ]) + 2.0

[xa1_den,xa1_x,xa1_y]=np.histogram2d(xf1_ens[0,:],xf1_ens[1,:],range=[[xmin,xmax],[ymin,ymax]],bins=50,weights=w,density=True )
#[xa0_den,xa0_x,xa0_y]=np.histogram2d(xa0_ens[0,:],xa0_ens[1,:],range=[[xmin,xmax],[ymin,ymax]], bins=50,density=True) 
[xf1_den,xf1_x,xf1_y]=np.histogram2d(xf1_ens[0,:],xf1_ens[1,:],range=[[xmin,xmax],[ymin,ymax]], bins=50,density=True) 

[xa1_xden,xbins] = np.histogram( xf1_ens[0,:],range=(xmin,xmax),bins=50,weights=w,density=True )
[xa1_yden,ybins] = np.histogram( xf1_ens[1,:],range=(ymin,ymax),bins=50,weights=w,density=True )
[xa1_zden,zbins] = np.histogram( xf1_ens[2,:],range=(zmin,zmax),bins=50,weights=w,density=True )

[xf1_xden,nada] = np.histogram( xf1_ens[0,:],range=(xmin,xmax),bins=50,density=True )
[xf1_yden,nada] = np.histogram( xf1_ens[1,:],range=(ymin,ymax),bins=50,density=True )
[xf1_zden,nada] = np.histogram( xf1_ens[2,:],range=(zmin,zmax),bins=50,density=True )

[xa0_xden,nada] = np.histogram( xa0_ens[0,:],range=(xmin,xmax),bins=50,density=True )
[xa0_yden,nada] = np.histogram( xa0_ens[1,:],range=(ymin,ymax),bins=50,density=True )
[xa0_zden,nada] = np.histogram( xa0_ens[2,:],range=(zmin,zmax),bins=50,density=True )

xplot = 0.5*( xbins[0:-1] + xbins[1:] )
yplot = 0.5*( ybins[0:-1] + ybins[1:] )
zplot = 0.5*( zbins[0:-1] + zbins[1:] )

#Likelihood computation
like_den = np.zeros( (xplot.size , yplot.size) )
for ii in range( xplot.size ) :
    for jj in range( yplot.size ) :
        #Nota, en este caso la likelihood es aproximada no es la marginalizacion de la likelihood en x e y. 
        like_den[ii,jj] = np.exp( -2.0 * np.sum( ( forward_operator( np.array([xplot[ii],yplot[jj],0]) ) - yo1 ) ** 2 ) )

metrics=dict()
[xfda_mean , xfda_cov ]=mean_covar( x_ens_evol[:,:,0] )
[xada_mean , xada_cov ]=mean_covar( x_ens_evol[:,:,-1] )
metrics['dist_covar_a']=dist_covar( xada_cov , xa1_cov )
metrics['dist_covar_f']=dist_covar( xfda_cov , xf1_cov )
metrics['dist_var_a']=dist_var( xada_cov , xa1_cov )
metrics['dist_var_f']=dist_var( xfda_cov , xf1_cov )
metrics['bias_var_a']=bias_var( xada_cov , xa1_cov )
metrics['bias_var_f']=bias_var( xfda_cov , xf1_cov )
metrics['kld_a']=kld3d(  x_ens_evol[:,:,-1] , xa1_den , xbins , ybins , zbins , xa1_mean , xa1_cov , 50 , adjust_mean=True )
metrics['kld_f']=kld3d(  x_ens_evol[:,:,0] , xf1_den , xbins , ybins , zbins , xf1_mean , xf1_cov , 50 , adjust_mean=True )
metrics['bias_f']=bias_mean( xfda_mean , xf1_mean )
metrics['bias_a']=bias_mean( xfda_mean , xf1_mean )
metrics['rmse_f']=rmse_mean( xfda_mean , xf1_mean )
metrics['rmse_a']=rmse_mean( xfda_mean , xf1_mean )

plot_xyz_evol( x_ens_evol , xplot , yplot , zplot , xf1_den , xf1_xden , xf1_yden , xf1_zden , xa1_den , xa1_xden , xa1_yden , xa1_zden , 
                               like_den , xt1 , yo1 , 'ETKF ONE STEP' , 'Figure_update_ETKF_1step_nonlinearsum_largeens_' ,  bst , numtrans , metrics )

        
#------------------------------------------------------------------------------
#  GM-ETPF
#------------------------------------------------------------------------------
bridge_parameter =0.5
sample_size = 500
#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol=np.zeros((nvars,EnsSize,2)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.

tmp_for = np.copy( xf1_ens_da )

R_loc = R / (1.0 - bridge_parameter )
[ tmp_for , tmp_anal_mean , Pa , null_var , null_var , wgm , kernel_perts ] =da.analysis_update_GM( yo1 , tmp_for , forward_operator , R_loc , 1.0 , beta_param = 0.6 , gamma_param = 0.0 )
R_loc = R / (bridge_parameter)
[tmp_anal , tmp_anal_mean , Pa , null_var , nul_var , tmp_sample ]= da.analysis_update_GMETPF( yo1 , tmp_for , wgm , kernel_perts , sample_size , forward_operator , R_loc  ) 

x_ens_evol[:,:,1] = tmp_anal

plt.figure()
plt.plot( np.transpose(x_ens_evol[0,:,:]) , np.transpose( x_ens_evol[1,:,:] ) , '--',color='#C0C0C0',linewidth=0.5,zorder=1)

plt.plot(x_ens_evol[0,:,0],x_ens_evol[1,:,0],'ro',label='Prior Ens',zorder=2,alpha=0.5)  #Prior
#plt.plot(tmp_for[0,:],tmp_for[1,:],'go',label='Intermediate Ens.',zorder=3,alpha=0.5)  #Posterior
#plt.plot(tmp_sample[0,:],tmp_sample[1,:],'yo',label='Intermediate Ens.',zorder=3,alpha=0.5)  #Posterior
plt.plot(x_ens_evol[0,:,-1],x_ens_evol[1,:,-1],'bo',label='Posterior Ens',zorder=3,alpha=0.5)  #Posterior
plt.plot(xt1[0],xt1[1],'ks',label='True')

denmax=np.max(xf1_den)
dden=denmax/3.0
levels=np.arange(0.001,denmax,dden)
plt.contour(xplot,yplot,np.transpose(xf1_den),colors=('#842020'),linewidths=3.0,levels=levels,zorder=5)
denmax=np.max(xa1_den)
dden=denmax/3.0
levels=np.arange(0.001,denmax,dden)
plt.contour(xplot,yplot,np.transpose(xa1_den),colors=('#000066'),linewidths=3.0,levels=levels,zorder=6)
denmax=np.max(like_den)
dden=denmax/3.0
levels=np.arange(0.001,denmax,dden)
plt.contour(xplot,yplot,np.transpose(like_den),colors='k',linewidths=2.0,levels=levels,zorder=4)

plot_xyz_evol( x_ens_evol , xplot , yplot , zplot , xf1_den , xf1_xden , xf1_yden , xf1_zden , xa1_den , xa1_xden , xa1_yden , xa1_zden , 
                               like_den , xt1 , yo1 , 'GM-LETPF' , 'Figure_update_GMLETPF_1step_nonlinearsum_' ,  bst , numtrans , metrics )


#------------------------------------------------------------------------------
#  GMDR en un paso
#------------------------------------------------------------------------------
beta_param=0.6
gamma_param=0.2

#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol=np.zeros((nvars,EnsSize,2)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.

tmp_for = np.copy( xf1_ens_da ) 
[ tmp_anal , tmp_anal_mean , null_var , null_var , null_var ] =da.analysis_update_GMDR( yo1 , tmp_for , forward_operator , R , 1.0 , beta_param , gamma_param  )


x_ens_evol[:,:,1] = tmp_anal

metrics=dict()
[xfda_mean , xfda_cov ]=mean_covar( x_ens_evol[:,:,0] )
[xada_mean , xada_cov ]=mean_covar( x_ens_evol[:,:,-1] )
metrics['dist_covar_a']=dist_covar( xada_cov , xa1_cov )
metrics['dist_covar_f']=dist_covar( xfda_cov , xf1_cov )
metrics['dist_var_a']=dist_var( xada_cov , xa1_cov )
metrics['dist_var_f']=dist_var( xfda_cov , xf1_cov )
metrics['bias_var_a']=bias_var( xada_cov , xa1_cov )
metrics['bias_var_f']=bias_var( xfda_cov , xf1_cov )
metrics['kld_a']=kld3d(  x_ens_evol[:,:,-1] , xa1_den , xbins , ybins , zbins , xa1_mean , xa1_cov , 50 , adjust_mean=True )
metrics['kld_f']=kld3d(  x_ens_evol[:,:,0] , xf1_den , xbins , ybins , zbins , xf1_mean , xf1_cov , 50 , adjust_mean=True )
metrics['bias_f']=bias_mean( xfda_mean , xf1_mean )
metrics['bias_a']=bias_mean( xada_mean , xa1_mean )
metrics['rmse_f']=rmse_mean( xfda_mean , xf1_mean )
metrics['rmse_a']=rmse_mean( xada_mean , xa1_mean )

plot_xyz_evol( x_ens_evol , xplot , yplot , zplot , xf1_den , xf1_xden , xf1_yden , xf1_zden , xa1_den , xa1_xden , xa1_yden , xa1_zden ,
                                like_den , xt1 , yo1 , 'GMDR ONE STEP' , 'Figure_update_GMDR_1step_nonlinearsum_largeens_' ,  bst , numtrans , metrics )






