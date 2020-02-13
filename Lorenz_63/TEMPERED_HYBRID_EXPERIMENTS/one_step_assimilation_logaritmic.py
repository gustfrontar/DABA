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

SampleSize=xa0_ens.shape[1]


#First chose a random member which will be the true. We can chose the last one insted of a random one.

xt0 = xa0_ens[:,-1]
xt1 = xf1_ens[:,-1]    #No model error

#Second generate an observation from the true
yo1 = forward_operator( xt1 ) + np.random.multivariate_normal(np.zeros(nobs),R*np.eye(nobs))
 
#Third compute the full posterior density using importance sampling
w=np.zeros(xa0_ens.shape[1])
yf1_ens = forward_operator_ens( xf1_ens )
dep = np.tile(yo1,(SampleSize,1)).T - yf1_ens[:]
if nobs == 1 :
   w=  np.exp( -np.power( dep , 2 )  / (2.0*R0) )  #This is a simplified version of the computation of the likelihood.
else         :
   w=  np.exp( np.sum( -np.power( dep , 2 ) , 0 )  / (2.0*R0) )  #This is a simplified version of the computation of the likelihood.    
w = w / np.sum(w) #Weigth normalization.

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
x_ens_evol_etkf=np.zeros((nvars,EnsSize,2)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_etkf[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.

tmp_for = np.copy( xf1_ens_da )
[ tmp_anal , tmp_anal_mean , Pa , null_var , null_var ] =da.analysis_update_ETKF( yo1 , tmp_for , forward_operator , R , 1.0 )

x_ens_evol_etkf[:,:,1] = tmp_anal


#Calculamos las densidades en un rango rodeando al true.
xmin = np.min([np.min(xt1[0]),np.min(x_ens_evol_etkf[0,:,:]) ]) - 2.0
xmax = np.max([np.max(xt1[0]),np.max(x_ens_evol_etkf[0,:,:]) ]) + 2.0
ymin = np.min([np.min(xt1[1]),np.min(x_ens_evol_etkf[1,:,:]) ]) - 2.0
ymax = np.max([np.max(xt1[1]),np.max(x_ens_evol_etkf[1,:,:]) ]) + 2.0
zmin = np.min([np.min(xt1[2]),np.min(x_ens_evol_etkf[2,:,:]) ]) - 2.0
zmax = np.max([np.max(xt1[2]),np.max(x_ens_evol_etkf[2,:,:]) ]) + 2.0

[xa1_den,xa1_x,xa1_y]=np.histogram2d(xf1_ens[0,:],xf1_ens[1,:],range=[[xmin,xmax],[ymin,ymax]],bins=50,weights=w,density=True )
#[xa0_den,xa0_x,xa0_y]=np.histogram2d(xa0_ens[0,:],xa0_ens[1,:],range=[[xmin,xmax],[ymin,ymax]], bins=50,density=True) 
[xf1_den,xf1_x,xf1_y]=np.histogram2d(xf1_ens[0,:],xf1_ens[1,:],range=[[xmin,xmax],[ymin,ymax]], bins=50,density=True) 

[xa1_xden,xa1_x] = np.histogram( xf1_ens[0,:],range=(xmin,xmax),bins=50,weights=w,density=True )
[xa1_yden,xa1_y] = np.histogram( xf1_ens[1,:],range=(ymin,ymax),bins=50,weights=w,density=True )
[xa1_zden,xa1_z] = np.histogram( xf1_ens[2,:],range=(zmin,zmax),bins=50,weights=w,density=True )

[xf1_xden,xf1_x] = np.histogram( xf1_ens[0,:],range=(xmin,xmax),bins=50,density=True )
[xf1_yden,xf1_y] = np.histogram( xf1_ens[1,:],range=(ymin,ymax),bins=50,density=True )
[xf1_zden,xf1_z] = np.histogram( xf1_ens[2,:],range=(zmin,zmax),bins=50,density=True )

[xa0_xden,xa0_x] = np.histogram( xa0_ens[0,:],range=(xmin,xmax),bins=50,density=True )
[xa0_yden,xa0_y] = np.histogram( xa0_ens[1,:],range=(ymin,ymax),bins=50,density=True )
[xa0_zden,xa0_z] = np.histogram( xa0_ens[2,:],range=(zmin,zmax),bins=50,density=True )

xa0_x = 0.5*( xa0_x[0:-1] + xa0_x[1:] )
xa0_y = 0.5*( xa0_y[0:-1] + xa0_y[1:] )
xa0_z = 0.5*( xa0_z[0:-1] + xa0_z[1:] )
xf1_x = 0.5*( xf1_x[0:-1] + xf1_x[1:] )
xf1_y = 0.5*( xf1_y[0:-1] + xf1_y[1:] )
xf1_z = 0.5*( xf1_z[0:-1] + xf1_z[1:] )
xa1_x = 0.5*( xa1_x[0:-1] + xa1_x[1:] )
xa1_y = 0.5*( xa1_y[0:-1] + xa1_y[1:] )
xa1_z = 0.5*( xa1_z[0:-1] + xa1_z[1:] )

#Likelihood computation
like_den = np.zeros( (xf1_x.size , xf1_y.size) )
for ii in range( xa1_x.size ) :
    for jj in range( xa1_y.size ) :
        #Nota, en este caso la likelihood es aproximada no es la marginalizacion de la likelihood en x e y. 
        like_den[ii,jj] = np.exp( -2.0 * np.sum( ( forward_operator( np.array([xa1_x[ii],xa1_y[jj],0]) ) - yo1 ) ** 2 ) )
        
plt.figure()
#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_etkf[0,:,0],x_ens_evol_etkf[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_etkf[0,:,1],x_ens_evol_etkf[1,:,1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_etkf[0,:,:]) , np.transpose( x_ens_evol_etkf[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Prior Den')
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Posterior Den')
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.legend()
plt.title('ETKF ONE STEP')
plt.savefig('Figure_update_ETKF_1step_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_etkf.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_etkf[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etkf[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etkf[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_etkf[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etkf[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etkf[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_etkf[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etkf[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etkf[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.legend()
plt.suptitle('ETKF ONE STEP')
plt.savefig('Figure_marginal_dist_ETKF_1step_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')


#------------------------------------------------------------------------------
#  ETKF con TEMPERADO
#------------------------------------------------------------------------------

#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_etkft=np.zeros((nvars,EnsSize,NTemp+1)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_etkft[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.


tmp_for = np.copy( xf1_ens_da ) 
Rt = R * NTemp
for itemp in range(NTemp) :
    [ tmp_for , tmp_anal_mean , Pa , null_var , null_var ] =da.analysis_update_ETKF( yo1 , tmp_for , forward_operator , Rt , 1.0 )

    x_ens_evol_etkft[:,:,itemp+1] = tmp_for

plt.figure()
#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_etkft[0,:,0],x_ens_evol_etkft[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_etkft[0,:,-1],x_ens_evol_etkft[1,:,-1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_etkft[0,:,:]) , np.transpose( x_ens_evol_etkft[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.title('ETKF TEMPERING')
plt.legend()

plt.savefig('Figure_update_ETKF_tempering_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_etkft.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_etkft[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etkft[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etkft[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_etkft[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etkft[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etkft[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_etkft[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etkft[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etkft[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.suptitle('ETKF TEMPERING')
plt.legend()
plt.savefig('Figure_marginal_dist_ETKF_tempering_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')


#------------------------------------------------------------------------------
#  ETKF con TEMPERADO EXPONENCIAL
#------------------------------------------------------------------------------

#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_etkft=np.zeros((nvars,EnsSize,NTemp+1)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_etkft[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.
tmp_for = np.copy(xf1_ens_da) 

alfa =1.0 / np.flip( np.exp( np.arange( 0 , NTemp ) ** 1.2 ) )
alfa = alfa / np.sum(alfa)
for itemp in range(NTemp) :
    Rt = R / alfa[itemp]
    [ tmp_for , tmp_anal_mean , Pa , null_var , null_var ] =da.analysis_update_ETKF( yo1 , tmp_for , forward_operator , Rt , 1.0 )
    x_ens_evol_etkft[:,:,itemp+1] = tmp_for
    print(np.sqrt(np.sum(np.power( x_ens_evol_etkft[:,:,itemp] - x_ens_evol_etkft[:,:,itemp+1] , 2 ) ) ))

plt.figure()
#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_etkft[0,:,0],x_ens_evol_etkft[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_etkft[0,:,-1],x_ens_evol_etkft[1,:,-1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_etkft[0,:,:]) , np.transpose( x_ens_evol_etkft[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.title('ETKF TEMPERING EXPONENTIAL')
plt.legend()

plt.savefig('Figure_update_ETKF_temperingexp_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_etkft.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_etkft[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etkft[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etkft[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_etkft[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etkft[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etkft[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_etkft[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etkft[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etkft[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.legend()
plt.suptitle('ETKF TEMPERING EXPONENTIAL')
plt.savefig('Figure_marginal_dist_ETKF_temperingexp_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')



#------------------------------------------------------------------------------
#  ETKF con RIP
#------------------------------------------------------------------------------

#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_etkfr=np.zeros((nvars,EnsSize,NRip+1)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_etkfr[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.
tmp_ens = np.zeros((nvars,EnsSize,2))
 
tmp_ens[:,:,0] = np.copy( xa0_ens_da )
Rt = R * NTemp
for irip in range(NRip) :
    
    tmp_ens[:,:,1] = model.forward_model( ne=EnsSize , x0=tmp_ens[:,:,0] , p=p , nt=bst , dt=dt  )
        
    [ tmp_ens[:,:,1] , null_var , null_var , null_var , null_var , tmp_ens[:,:,0] ] =da.analysis_update_ETKF_rip( yo1 , tmp_ens , forward_operator , Rt , 1.0 )

    x_ens_evol_etkfr[:,:,irip+1] = tmp_ens[:,:,1]
    

plt.figure()

#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_etkfr[0,:,0],x_ens_evol_etkfr[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_etkfr[0,:,-1],x_ens_evol_etkfr[1,:,-1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_etkfr[0,:,:]) , np.transpose( x_ens_evol_etkfr[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.legend()
plt.title('ETKF RIP')
plt.savefig('Figure_update_ETKF_rip_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')


#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_etkfr.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_etkfr[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etkfr[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etkfr[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_etkfr[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etkfr[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etkfr[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_etkfr[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etkfr[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etkfr[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.legend()
plt.suptitle('ETKF RIP')
plt.savefig('Figure_marginal_dist_ETKF_rip_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')


#------------------------------------------------------------------------------
#  ETKF con RIP EXPONENCIAL
#------------------------------------------------------------------------------

#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_etkfr=np.zeros((nvars,EnsSize,NRip+1)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_etkfr[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.
tmp_ens = np.zeros((nvars,EnsSize,2))
 
tmp_ens[:,:,0] = np.copy( xa0_ens_da )
Rt = R * NTemp


alfa =1.0 / np.flip( np.exp( np.arange( 0 , NRip ) ** 1.2 ) )
alfa = alfa / np.sum(alfa)
for irip in range(NRip) :
    Rt = R / alfa[irip]
    tmp_ens[:,:,1] = model.forward_model( ne=EnsSize , x0=tmp_ens[:,:,0] , p=p , nt=bst , dt=dt  )
        
    [ tmp_ens[:,:,1] , null_var , null_var , null_var , null_var , tmp_ens[:,:,0] ] =da.analysis_update_ETKF_rip( yo1 , tmp_ens , forward_operator , Rt , 1.0 )

    x_ens_evol_etkfr[:,:,irip+1] = tmp_ens[:,:,1]
    

plt.figure()

#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_etkfr[0,:,0],x_ens_evol_etkfr[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_etkfr[0,:,-1],x_ens_evol_etkfr[1,:,-1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_etkfr[0,:,:]) , np.transpose( x_ens_evol_etkfr[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.legend()
plt.title('ETKF RIP EXPONENTIAL')
plt.savefig('Figure_update_ETKF_ripexp_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')


#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_etkfr.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_etkfr[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etkfr[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etkfr[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_etkfr[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etkfr[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etkfr[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_etkfr[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etkfr[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etkfr[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.legend()
plt.suptitle('ETKF RIP EXPONENTIAL')
plt.savefig('Figure_marginal_dist_ETKF_ripexp_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')


#------------------------------------------------------------------------------
#  ETKF con RIP ADAPTATIVO
#------------------------------------------------------------------------------

#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_etkfr=np.zeros((nvars,EnsSize,NRip+1)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_etkfr[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.
tmp_ens = np.zeros((nvars,EnsSize,2))
 
tmp_ens[:,:,0] = np.copy( xa0_ens_da )

target_increment = 1.0 / NRip   #Adapt the tempering step so the increment in each iteration is almost constant (no sure if this is the optimal solution)
y=np.zeros((nobs,EnsSize))

#Compute HPHt
for iens in range( EnsSize ) :
    y[:,iens] = forward_operator( tmp_ens[:,iens,0] )
if nobs > 1 :
   temp_coef = 1.0 / ( target_increment ) + ( ( 1.0 - target_increment ) / target_increment ) * np.trace( np.cov( y ) ) / np.trace( R )
else        :
   temp_coef = 1.0 / ( target_increment ) + ( ( 1.0 - target_increment ) / target_increment ) * np.cov( y )  /  R 

dt_1 = 1.0/temp_coef 
#Asumo pasos de tiempo que se incrementan linealmente con el pseudo tiempo.
b_dt =( 1.0 - NRip * dt_1 ) / ( 0.5*NRip * (NRip +1.0)  - NRip  )
a_dt = dt_1 - b_dt

pseudo_time = 0.0  #Initialize pseudo time.
for irip in range(NRip) :

    dt_rip = a_dt + b_dt * ( 1.0 + irip )
    Rt = R * ( 1.0 / dt_rip )
    pseudo_time = pseudo_time + dt_rip 
    
    print(temp_coef , pseudo_time )
    tmp_ens[:,:,1] = model.forward_model( ne=EnsSize , x0=tmp_ens[:,:,0] , p=p , nt=bst , dt=dt  )
        
    [ tmp_ens[:,:,1] , null_var , null_var , null_var , null_var , tmp_ens[:,:,0] ] =da.analysis_update_ETKF_rip( yo1 , tmp_ens , forward_operator , Rt , 1.0 )

    x_ens_evol_etkfr[:,:,irip+1] = tmp_ens[:,:,1]

plt.figure()

#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_etkfr[0,:,0],x_ens_evol_etkfr[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_etkfr[0,:,-1],x_ens_evol_etkfr[1,:,-1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_etkfr[0,:,:]) , np.transpose( x_ens_evol_etkfr[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.legend()
plt.title('ETKF RIP ADDAPTIVE')
plt.savefig('Figure_update_ETKF_ripaddaptive_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')


#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_etkfr.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.suptitle('ETKF RIP ADDAPTIVE')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_etkfr[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etkfr[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etkfr[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_etkfr[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etkfr[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etkfr[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_etkfr[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etkfr[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etkfr[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.legend()
plt.savefig('Figure_marginal_dist_ETKF_ripaddaptive_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')


#------------------------------------------------------------------------------
#  ETPF en un paso
#------------------------------------------------------------------------------

#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_etpf=np.zeros((nvars,EnsSize,2)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_etpf[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.

tmp_for = np.copy( xf1_ens_da ) 
[ tmp_anal , tmp_anal_mean , Pa , null_var , null_var , w , null_var ] =da.analysis_update_ETPF_2ndord( yo1 , tmp_for , forward_operator , R , 0.0 , 0.0  )

x_ens_evol_etpf[:,:,1] = tmp_anal

plt.figure()
#plt.plot(yo1[0],yo1[1],'ks')
#plt.scatter(x_ens_evol_etpf[0,:,0],x_ens_evol_etpf[1,:,0],s=np.power(w,1.0/4.0)*100.0,c='r',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_etpf[0,:,0],x_ens_evol_etpf[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_etpf[0,:,1],x_ens_evol_etpf[1,:,1],'bo',label='Posterior Ens',markersize=4.0)  #Posterior
plt.plot( np.transpose(x_ens_evol_etpf[0,:,:]) , np.transpose( x_ens_evol_etpf[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Prior Den')
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Posterior Den')
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.title('ETPF ONE STEP')
plt.legend()

plt.savefig('Figure_update_ETPF_1step_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_etpf.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_etpf[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etpf[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etpf[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_etpf[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etpf[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etpf[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_etpf[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etpf[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etpf[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.suptitle('ETPF ONE STEP')
plt.legend()
plt.savefig('Figure_marginal_dist_ETPF_1step_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#------------------------------------------------------------------------------
#  ETPF con temperado
#------------------------------------------------------------------------------

#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_etpft=np.zeros((nvars,EnsSize,NTemp+1)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_etpft[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.

tmp_ftmp_for = np.copy( xf1_ens_da )
Rt = R * NTemp
for itemp in range(NTemp) : 
    [ tmp_for , tmp_anal_mean , Pa , null_var , null_var , null_var , null_var ] =da.analysis_update_ETPF_2ndord( yo1 , tmp_for , forward_operator , Rt , 0.0 , 0.0  )
    x_ens_evol_etpft[:,:,itemp+1] = tmp_for


plt.figure()
#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_etpft[0,:,0],x_ens_evol_etpft[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_etpft[0,:,-1],x_ens_evol_etpft[1,:,-1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_etpft[0,:,:]) , np.transpose( x_ens_evol_etpft[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.title('ETPF TEMPERING')
plt.legend()

plt.savefig('Figure_update_ETPF_tempering_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_etpft.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_etpft[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etpft[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etpft[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_etpft[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etpft[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etpft[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_etpft[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etpft[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etpft[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.suptitle('ETPF TEMPERING')
plt.legend()
plt.savefig('Figure_marginal_dist_ETPF_tempering_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')



#------------------------------------------------------------------------------
#  ETPF con temperado con pasos exponenciales
#------------------------------------------------------------------------------

#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_etpft=np.zeros((nvars,EnsSize,NTemp+1)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_etpft[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.

tmp_ftmp_for = np.copy( xf1_ens_da )
alfa =1.0 / np.flip( np.exp( np.arange( 0 , NTemp ) ** 1.2 ) )
alfa = alfa / np.sum(alfa)

for itemp in range(NTemp) :
    Rt = R / alfa[itemp] 
    [ tmp_for , tmp_anal_mean , Pa , null_var , null_var , null_var , null_var ] =da.analysis_update_ETPF_2ndord( yo1 , tmp_for , forward_operator , Rt , 0.0 , 0.0  )
    x_ens_evol_etpft[:,:,itemp+1] = tmp_for


plt.figure()
#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_etpft[0,:,0],x_ens_evol_etpft[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_etpft[0,:,-1],x_ens_evol_etpft[1,:,-1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_etpft[0,:,:]) , np.transpose( x_ens_evol_etpft[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.title('ETPF TEMPERING EXPONENTIAL')
plt.legend()

plt.savefig('Figure_update_ETPF_temperingexp_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_etpft.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_etpft[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etpft[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etpft[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_etpft[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etpft[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etpft[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_etpft[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etpft[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etpft[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.suptitle('ETPF TEMPERING EXPONENTIAL')
plt.legend()
plt.savefig('Figure_marginal_dist_ETPF_temperingexp_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')



#------------------------------------------------------------------------------
#  ETPF con RIP
#------------------------------------------------------------------------------

#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_etpfr=np.zeros((nvars,EnsSize,NRip+1)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_etpfr[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.
tmp_ens = np.zeros((nvars,EnsSize,2))
 
tmp_ens[:,:,0] = np.copy( xa0_ens_da )
Rt = R * NRip
for irip in range(NRip) :
    
    tmp_ens[:,:,1] = model.forward_model( ne=EnsSize , x0=tmp_ens[:,:,0] , p=p , nt=bst , dt=dt  )
    [ tmp_ens[:,:,1] , null_var , null_var , tmp_ens[:,:,0 ] ] =da.analysis_update_ETPF_2ndord_rip( yo1 , tmp_ens , forward_operator , Rt , 0.0 , 0.0  )

    x_ens_evol_etpfr[:,:,irip+1] = tmp_ens[:,:,1]
    

plt.figure()

#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_etpfr[0,:,0],x_ens_evol_etpfr[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_etpfr[0,:,-1],x_ens_evol_etpfr[1,:,-1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_etpfr[0,:,:]) , np.transpose( x_ens_evol_etpfr[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.legend()
plt.title('ETPF RIP')

plt.savefig('Figure_update_ETPF_rip_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_etpfr.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_etpfr[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etpfr[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etpfr[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_etpfr[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etpfr[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etpfr[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_etpfr[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etpfr[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etpfr[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.legend()
plt.suptitle('ETPF RIP')
plt.savefig('Figure_marginal_dist_ETPF_rip_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')


#------------------------------------------------------------------------------
#  ETPF con RIP EXPONENCIAL
#------------------------------------------------------------------------------

#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_etpfr=np.zeros((nvars,EnsSize,NRip+1)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_etpfr[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.
tmp_ens = np.zeros((nvars,EnsSize,2))
 
tmp_ens[:,:,0] = np.copy( xa0_ens_da )
alfa =1.0 / np.flip( np.exp( np.arange( 0 , NRip ) ** 1.2 ) )
alfa = alfa / np.sum(alfa)

for irip in range(NRip) :
    Rt = R / alfa[irip] 
    tmp_ens[:,:,1] = model.forward_model( ne=EnsSize , x0=tmp_ens[:,:,0] , p=p , nt=bst , dt=dt  )
    [ tmp_ens[:,:,1] , null_var , null_var , tmp_ens[:,:,0 ] ] =da.analysis_update_ETPF_2ndord_rip( yo1 , tmp_ens , forward_operator , Rt , 0.0 , 0.0  )

    x_ens_evol_etpfr[:,:,irip+1] = tmp_ens[:,:,1]
    

plt.figure()

#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_etpfr[0,:,0],x_ens_evol_etpfr[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_etpfr[0,:,-1],x_ens_evol_etpfr[1,:,-1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_etpfr[0,:,:]) , np.transpose( x_ens_evol_etpfr[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.legend()
plt.title('ETPF RIP EXPONENTIAL')

plt.savefig('Figure_update_ETPF_ripexp_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_etpfr.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_etpfr[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etpfr[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etpfr[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_etpfr[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etpfr[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etpfr[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_etpfr[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_etpfr[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_etpfr[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.legend()
plt.suptitle('ETPF RIP EXPONENTIAL')
plt.savefig('Figure_marginal_dist_ETPF_ripexp_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')



#------------------------------------------------------------------------------
#  HIBRIDO ETKF - ETPF  EN UN SOLO PASO  BRIDGE PARAMETER 0.5
#------------------------------------------------------------------------------
bridge_param = 0.5
#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_hib=np.zeros((nvars,EnsSize,3)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_hib[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.

tmp_for = np.copy( xf1_ens_da ) 
Rh = R / ( 1.0 - bridge_param )
[ tmp_for , tmp_anal_mean , Pa , null_var , null_var ] =da.analysis_update_ETKF( yo1 , tmp_for , forward_operator , Rh , 1.0 )
x_ens_evol_hib[:,:,1] = tmp_for
Rh = R / ( bridge_param )
[ tmp_anal , null_var , null_var , null_var , null_var , w , null_var ] =da.analysis_update_ETPF_2ndord( yo1 , tmp_for , forward_operator , Rh , 0.0 , 0.0  )
x_ens_evol_hib[:,:,2] = tmp_anal

plt.figure()
#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_hib[0,:,0],x_ens_evol_hib[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_hib[0,:,-1],x_ens_evol_hib[1,:,-1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_hib[0,:,:]) , np.transpose( x_ens_evol_hib[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Prior Den')
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Posterior Den')
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.title('HIBRID ONE STEP')
plt.legend()

plt.savefig('Figure_update_HIB_1step_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_hib.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_hib[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_hib[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_hib[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_hib[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_hib[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_hib[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_hib[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_hib[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_hib[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.legend()
plt.suptitle('HIBRID ONE STEP')
plt.savefig('Figure_marginal_dist_HIB_1step_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')



#------------------------------------------------------------------------------
#  HIBRIDO ETKF - ETPF  CON TEMPERADO   BRIDGE PARAMETER 0.5
#------------------------------------------------------------------------------
bridge_param = 0.5
#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_hibt=np.zeros((nvars,EnsSize,1+2*NTemp)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_hibt[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.

tmp_for = np.copy( xf1_ens_da ) 
alfa = np.ones(NTemp) / NTemp
for itemp in range(NTemp)  :
   Rh = R / ( alfa[itemp] * (1.0 - bridge_param ) ) 
   [ tmp_for , tmp_anal_mean , Pa , null_var , null_var ] =da.analysis_update_ETKF( yo1 , tmp_for , forward_operator , Rh , 1.0 )
   x_ens_evol_hibt[:,:,itemp*2+1] = tmp_for
   Rh = R / ( alfa[itemp] * ( bridge_param ) ) 
   [ tmp_for , null_var , null_var , null_var , null_var , w , null_var ] =da.analysis_update_ETPF_2ndord( yo1 , tmp_for , forward_operator , Rh , 0.0 , 0.0  )
   x_ens_evol_hibt[:,:,itemp*2+2] = tmp_for

plt.figure()
#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_hibt[0,:,0],x_ens_evol_hibt[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_hibt[0,:,-1],x_ens_evol_hibt[1,:,-1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_hibt[0,:,:]) , np.transpose( x_ens_evol_hibt[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Prior Den')
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Posterior Den')
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.title('HIBRID TEMPERING')
plt.legend()

plt.savefig('Figure_update_HIB_tempering_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_hibt.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_hibt[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_hibt[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_hibt[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_hibt[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_hibt[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_hibt[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_hibt[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_hibt[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_hibt[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.legend()
plt.suptitle('HIBRID TEMPERING')
plt.savefig('Figure_marginal_dist_HIB_tempering_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')


#------------------------------------------------------------------------------
#  HIBRIDO ETKF - ETPF  CON TEMPERADO EXPONENCIAL BRIDGE PARAMETER 0.5
#------------------------------------------------------------------------------
bridge_param = 0.5
#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_hibt=np.zeros((nvars,EnsSize,1+2*NTemp)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_hibt[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.

tmp_for = np.copy( xf1_ens_da ) 
alfa =1.0 / np.flip( np.exp( np.arange( 0 , NTemp ) ** 1.2 ) )
alfa = alfa / np.sum(alfa)
for itemp in range(NTemp)  :
   Rh = R / ( alfa[itemp] * (1.0 - bridge_param ) ) 
   [ tmp_for , tmp_anal_mean , Pa , null_var , null_var ] =da.analysis_update_ETKF( yo1 , tmp_for , forward_operator , Rh , 1.0 )
   x_ens_evol_hibt[:,:,itemp*2+1] = tmp_for
   Rh = R / ( alfa[itemp] * ( bridge_param ) )
   [ tmp_for , null_var , null_var , null_var , null_var , w , null_var ] =da.analysis_update_ETPF_2ndord( yo1 , tmp_for , forward_operator , Rh , 0.0 , 0.0  )
   x_ens_evol_hibt[:,:,itemp*2+2] = tmp_for

plt.figure()
#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_hibt[0,:,0],x_ens_evol_hibt[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_hibt[0,:,-1],x_ens_evol_hibt[1,:,-1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_hibt[0,:,:]) , np.transpose( x_ens_evol_hibt[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Prior Den')
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Posterior Den')
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.title('HIBRID TEMPERING EXPONENTIAL')
plt.legend()

plt.savefig('Figure_update_HIB_temperingexp_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_hibt.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_hibt[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_hibt[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_hibt[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_hibt[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_hibt[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_hibt[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_hibt[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_hibt[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_hibt[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.suptitle('HIBRID TEMPERING EXPONENTIAL')
plt.legend()
plt.savefig('Figure_marginal_dist_HIB_temperingexp_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#------------------------------------------------------------------------------
#  HIBRIDO ETKF - ETPF  CON RUNNING IN PLACE   BRIDGE PARAMETER 0.5
#------------------------------------------------------------------------------

bridge_param = 0.5
#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_hibr=np.zeros((nvars,EnsSize,1+2*NRip)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_hibr[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.

tmp_ens = np.zeros((nvars,EnsSize,2))
tmp_ens[:,:,0] = np.copy( xa0_ens_da )

for irip in range(NRip) :
   tmp_ens[:,:,1] = model.forward_model( ne=EnsSize , x0=tmp_ens[:,:,0] , p=p , nt=bst , dt=dt  )
   Rt = R * NRip / ( 1.0 - bridge_param )
   [ tmp_ens[:,:,1] , null_var , null_var , null_var , null_var , tmp_ens[:,:,0] ] =da.analysis_update_ETKF_rip( yo1 , tmp_ens , forward_operator , Rt , 1.0 )
   x_ens_evol_hibr[:,:,irip*2+1] = tmp_ens[:,:,1]
   Rt = R * NRip / ( bridge_param )
   [ tmp_ens[:,:,1] , null_var , null_var , tmp_ens[:,:,0 ] ] =da.analysis_update_ETPF_2ndord_rip( yo1 , tmp_ens , forward_operator , Rt , 0.0 , 0.0  )
   x_ens_evol_hibr[:,:,irip*2+2] = tmp_ens[:,:,1]

plt.figure()
#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_hibr[0,:,0],x_ens_evol_hibr[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_hibr[0,:,-1],x_ens_evol_hibr[1,:,-1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_hibr[0,:,:]) , np.transpose( x_ens_evol_hibr[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Prior Den')
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Posterior Den')
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.title('HIBRID RIP')
plt.legend()

plt.savefig('Figure_update_HIB_rip_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_hibr.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_hibr[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_hibr[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_hibr[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_hibr[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_hibr[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_hibr[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_hibr[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_hibr[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_hibr[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.legend()
plt.suptitle('HIBRID RIP')
plt.savefig('Figure_marginal_dist_HIB_rip_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#------------------------------------------------------------------------------
#  HIBRIDO ETKF - ETPF  CON RUNNING IN PLACE EXPONENCIAL  BRIDGE PARAMETER 0.5
#------------------------------------------------------------------------------

bridge_param = 0.5
#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_hibr=np.zeros((nvars,EnsSize,1+2*NRip)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_hibr[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.

tmp_ens = np.zeros((nvars,EnsSize,2))
tmp_ens[:,:,0] = np.copy( xa0_ens_da )

alfa =1.0 / np.flip( np.exp( np.arange( 0 , NRip ) ** 1.2 ) )
alfa = alfa / np.sum(alfa)
for irip in range(NRip)  :
   Rh = R / ( alfa[irip] * (1.0 - bridge_param ) ) 
   tmp_ens[:,:,1] = model.forward_model( ne=EnsSize , x0=tmp_ens[:,:,0] , p=p , nt=bst , dt=dt  )
   Rt = R * NRip / ( 1.0 - bridge_param )
   [ tmp_ens[:,:,1] , null_var , null_var , null_var , null_var , tmp_ens[:,:,0] ] =da.analysis_update_ETKF_rip( yo1 , tmp_ens , forward_operator , Rt , 1.0 )
   x_ens_evol_hibr[:,:,irip*2+1] = tmp_ens[:,:,1]
   Rh = R / ( alfa[irip] * ( bridge_param ) )
   [ tmp_ens[:,:,1] , null_var , null_var , tmp_ens[:,:,0 ] ] =da.analysis_update_ETPF_2ndord_rip( yo1 , tmp_ens , forward_operator , Rt , 0.0 , 0.0  )
   x_ens_evol_hibr[:,:,irip*2+2] = tmp_ens[:,:,1]

plt.figure()
#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_hibr[0,:,0],x_ens_evol_hibr[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_hibr[0,:,-1],x_ens_evol_hibr[1,:,-1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_hibr[0,:,:]) , np.transpose( x_ens_evol_hibr[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Prior Den')
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Posterior Den')
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.title('HIBRID RIP EXPONENTIAL')
plt.legend()

plt.savefig('Figure_update_HIB_ripexp_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_hibr.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_hibr[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_hibr[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_hibr[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_hibr[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_hibr[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_hibr[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_hibr[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_hibr[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_hibr[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.legend()
plt.suptitle('HIBRID RIP EXPONENTIAL')
plt.savefig('Figure_marginal_dist_HIB_ripexp_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')


#------------------------------------------------------------------------------
#  GMDR en un paso
#------------------------------------------------------------------------------
beta_param=0.6
gamma_param=0.2

#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_gmdr=np.zeros((nvars,EnsSize,2)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_gmdr[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.

tmp_for = np.copy( xf1_ens_da ) 
[ tmp_anal , tmp_anal_mean , null_var , null_var , null_var ] =da.analysis_update_GMDR( yo1 , tmp_for , forward_operator , R , 1.0 , beta_param , gamma_param  )


x_ens_evol_gmdr[:,:,1] = tmp_anal

plt.figure()
#plt.plot(yo1[0],yo1[1],'ks')
#plt.scatter(x_ens_evol_gmdr[0,:,0],x_ens_evol_gmdr[1,:,0],s=np.power(w,1.0/4.0)*100.0,c='r',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_gmdr[0,:,0],x_ens_evol_gmdr[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_gmdr[0,:,1],x_ens_evol_gmdr[1,:,1],'bo',label='Posterior Ens',markersize=4.0)  #Posterior
plt.plot( np.transpose(x_ens_evol_gmdr[0,:,:]) , np.transpose( x_ens_evol_gmdr[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Prior Den')
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Posterior Den')
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.title('GMDR ONE STEP')
plt.legend()

plt.savefig('Figure_update_GMDR_1step_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_gmdr.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_gmdr[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_gmdr[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_gmdr[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_gmdr[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_gmdr[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_gmdr[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_gmdr[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_gmdr[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_gmdr[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.suptitle('GMDR ONE STEP')
plt.legend()
plt.savefig('Figure_marginal_dist_GMDR_1step_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#------------------------------------------------------------------------------
#  GMDR con temperado
#------------------------------------------------------------------------------
beta_param=0.6
gamma_param=0.2
#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_gmdrt=np.zeros((nvars,EnsSize,NTemp+1)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_gmdrt[:,:,0] = np.copy(xf1_ens_da) #El tiempo 0 son las particulas sampleadas del prior.

tmp_for = np.copy( xf1_ens_da )
Rt = R * NTemp
for itemp in range(NTemp) : 
    
    [ tmp_for , tmp_anal_mean , null_var , null_var , null_var ] =da.analysis_update_GMDR( yo1 , tmp_for , forward_operator , Rt , 1.0 , beta_param , gamma_param  )
    x_ens_evol_gmdrt[:,:,itemp+1] = tmp_for


plt.figure()
#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_gmdrt[0,:,0],x_ens_evol_gmdrt[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_gmdrt[0,:,-1],x_ens_evol_gmdrt[1,:,-1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_gmdrt[0,:,:]) , np.transpose( x_ens_evol_gmdrt[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.title('GMDR TEMPERING')
plt.legend()

plt.savefig('Figure_update_GMDR_tempering_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_gmdrt.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_gmdrt[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_gmdrt[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_gmdrt[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_gmdrt[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_gmdrt[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_gmdrt[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_gmdrt[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_gmdrt[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_gmdrt[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.legend()
plt.suptitle('GMDR TEMPERING')
plt.savefig('Figure_marginal_dist_GMDR_tempering_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')



#------------------------------------------------------------------------------
#  GMDR con temperado con pasos exponenciales
#------------------------------------------------------------------------------
beta_param=0.6
gamma_param=0.2
#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_gmdrt=np.zeros((nvars,EnsSize,NTemp+1)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_gmdrt[:,:,0] = np.copy(xf1_ens_da) #El tiempo 0 son las particulas sampleadas del prior.

tmp_for = np.copy( xf1_ens_da )
alfa =1.0 / np.flip( np.exp( np.arange( 0 , NTemp ) ** 1.2 ) )
alfa = alfa / np.sum(alfa)

for itemp in range(NTemp) :
    Rt = R / alfa[itemp] 
    [ tmp_for , tmp_anal_mean , null_var , null_var , null_var ] =da.analysis_update_GMDR( yo1 , tmp_for , forward_operator , Rt , 1.0 , beta_param , gamma_param  )
    x_ens_evol_gmdrt[:,:,itemp+1] = tmp_for


plt.figure()
#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_gmdrt[0,:,0],x_ens_evol_gmdrt[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_gmdrt[0,:,-1],x_ens_evol_gmdrt[1,:,-1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_gmdrt[0,:,:]) , np.transpose( x_ens_evol_gmdrt[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.title('GMDR TEMPERING EXPONENTIAL')
plt.legend()

plt.savefig('Figure_update_GMDR_temperingexp_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_gmdrt.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_gmdrt[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_gmdrt[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_gmdrt[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_gmdrt[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_gmdrt[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_gmdrt[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_gmdrt[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_gmdrt[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_gmdrt[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.legend()
plt.suptitle('GMDR TEMPERING EXPONENTIAL')
plt.savefig('Figure_marginal_dist_GMDR_temperingexp_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')



#------------------------------------------------------------------------------
#  GMDR con RIP
#------------------------------------------------------------------------------

#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_gmdrr=np.zeros((nvars,EnsSize,NRip+1)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_gmdrr[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.
tmp_ens = np.zeros((nvars,EnsSize,2))
 
tmp_ens[:,:,0] = np.copy( xa0_ens_da )
Rt = R * NRip
for irip in range(NRip) :
    
    tmp_ens[:,:,1] = model.forward_model( ne=EnsSize , x0=tmp_ens[:,:,0] , p=p , nt=bst , dt=dt  )
    [ tmp_ens[:,:,1] , tmp_anal_mean , null_var , null_var , null_var , tmp_ens[:,:,0] ] =da.analysis_update_GMDR_rip( yo1 , tmp_ens , forward_operator , Rt , 1.0 , beta_param , gamma_param  )

    x_ens_evol_gmdrr[:,:,irip+1] = tmp_ens[:,:,1]
    

plt.figure()

#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_gmdrr[0,:,0],x_ens_evol_gmdrr[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_gmdrr[0,:,-1],x_ens_evol_gmdrr[1,:,-1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_gmdrr[0,:,:]) , np.transpose( x_ens_evol_gmdrr[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.title('GMDR RIP')
plt.legend()

plt.savefig('Figure_update_GMDR_rip_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_gmdrr.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_gmdrr[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_gmdrr[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_gmdrr[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_gmdrr[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_gmdrr[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_gmdrr[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_gmdrr[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_gmdrr[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_gmdrr[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.suptitle('GMDR RIP')
plt.legend()
plt.savefig('Figure_marginal_dist_GMDR_rip_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')


#------------------------------------------------------------------------------
#  GMDR con RIP EXPONENCIAL
#------------------------------------------------------------------------------

#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_gmdrr=np.zeros((nvars,EnsSize,NRip+1)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_gmdrr[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.
tmp_ens = np.zeros((nvars,EnsSize,2))
 
tmp_ens[:,:,0] = np.copy( xa0_ens_da )
alfa =1.0 / np.flip( np.exp( np.arange( 0 , NRip ) ** 1.2 ) )
alfa = alfa / np.sum(alfa)

for irip in range(NRip) :
    Rt = R / alfa[irip] 
    tmp_ens[:,:,1] = model.forward_model( ne=EnsSize , x0=tmp_ens[:,:,0] , p=p , nt=bst , dt=dt  )
    [ tmp_ens[:,:,1] , tmp_anal_mean , null_var , null_var , null_var , tmp_ens[:,:,0] ] =da.analysis_update_GMDR_rip( yo1 , tmp_ens , forward_operator , Rt , 1.0 , beta_param , gamma_param  )
    x_ens_evol_gmdrr[:,:,irip+1] = tmp_ens[:,:,1]
    

plt.figure()

#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_gmdrr[0,:,0],x_ens_evol_gmdrr[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_gmdrr[0,:,-1],x_ens_evol_gmdrr[1,:,-1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_gmdrr[0,:,:]) , np.transpose( x_ens_evol_gmdrr[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.title('GMDR RIP EXPONENTIAL')
plt.legend()

plt.savefig('Figure_update_GMDR_ripexp_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_gmdrr.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_gmdrr[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_gmdrr[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_gmdrr[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_gmdrr[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_gmdrr[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_gmdrr[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_gmdrr[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_gmdrr[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_gmdrr[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.suptitle('GMDR RIP EXPONENTIAL')
plt.legend()
plt.savefig('Figure_marginal_dist_GMDR_ripexp_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')


#------------------------------------------------------------------------------
#  GPF en un solo paso (Este metodo es esencialmente un temperado con lo cual
#  no tiene sentido probar temperado para este metodo pero si quizas RIP)
#------------------------------------------------------------------------------

#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
import Lorenz_63_DA as da
tmp_for = np.copy( xf1_ens_da ) 
[ tmp_anal , tmp_anal_mean , x_ens_evol_gpf ] = da.analysis_update_GPF( yo1 , tmp_for , forward_operator , forward_operator_tl , R )


plt.figure()
#plt.plot(yo1[0],yo1[1],'ks')
#plt.scatter(x_ens_evol_etpf[0,:,0],x_ens_evol_etpf[1,:,0],s=np.power(w,1.0/4.0)*100.0,c='r',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_gpf[0,:,0],x_ens_evol_gpf[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_gpf[0,:,-1],x_ens_evol_gpf[1,:,-1],'bo',label='Posterior Ens',markersize=4.0)  #Posterior
plt.plot( np.transpose(x_ens_evol_gpf[0,:,:]) , np.transpose( x_ens_evol_gpf[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x,xf1_y,np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Prior Den')
plt.contour(xa1_x,xa1_y,np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Posterior Den')
plt.contour(xa1_x,xa1_y,np.transpose(like_den),colors='k',linewidths=0.4,label='Likelihood Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.title('GPF')

plt.legend()

plt.savefig('Figure_update_GPF_1step_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')

#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
Ntimes = x_ens_evol_gpf.shape[2]
pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
plt.subplot(1,3,1)
plt.plot(x_ens_evol_gpf[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_gpf[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_gpf[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.title('X')
plt.plot(xt1[0],1,'ks',label='True')
plt.subplot(1,3,2)
plt.plot(x_ens_evol_gpf[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_gpf[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_gpf[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[1],1,'ks',label='True')
plt.title('Y')
plt.subplot(1,3,3)
plt.plot(x_ens_evol_gpf[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
plt.plot(x_ens_evol_gpf[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
plt.plot(x_ens_evol_gpf[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
plt.plot(xt1[2],1,'ks',label='True')
plt.title('Z')
plt.legend()
plt.suptitle('GPF')
plt.savefig('Figure_marginal_dist_GPF_1step_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')




