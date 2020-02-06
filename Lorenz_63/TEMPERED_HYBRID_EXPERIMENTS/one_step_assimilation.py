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
from Lorenz_63_ObsOperator import forward_operator_nonlinear    as forward_operator
from Lorenz_63_ObsOperator import forward_operator_nonlinear_ens    as forward_operator_ens

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

R0=1.5
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
dep = np.zeros(yf1_ens.shape)
for iv in range(nvars):
    dep[iv,:] = yo1[iv] - yf1_ens[iv,:]
w=  np.exp( np.sum( -np.power( dep , 2 ) , 0 ) / (2.0*R0) )  #This is a simplified version of the computation of the likelihood.
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

tmp_for = xf1_ens_da 
[ tmp_anal , tmp_anal_mean , Pa , null_var , null_var ] =da.analysis_update_ETKF( yo1 , tmp_for , forward_operator , R , 1.0 )

x_ens_evol_etkf[:,:,1] = tmp_anal


#Calculamos las densidades en un rango rodeando al true.
xmin = np.min([np.min(xt1[0]),np.min(x_ens_evol_etkf[0,:,:]) ]) - 2.0
xmax = np.max([np.max(xt1[0]),np.max(x_ens_evol_etkf[0,:,:]) ]) + 2.0
ymin = np.min([np.min(xt1[1]),np.min(x_ens_evol_etkf[1,:,:]) ]) - 2.0
ymax = np.max([np.max(xt1[1]),np.max(x_ens_evol_etkf[1,:,:]) ]) + 2.0

[xa1_den,xa1_x,xa1_y]=np.histogram2d(xf1_ens[0,:],xf1_ens[1,:],range=[[xmin,xmax],[ymin,ymax]], bins=50,weights=w ,density=True)
[xa0_den,xa0_x,xa0_y]=np.histogram2d(xa0_ens[0,:],xa0_ens[1,:],range=[[xmin,xmax],[ymin,ymax]], bins=50,density=True) 
[xf1_den,xf1_x,xf1_y]=np.histogram2d(xf1_ens[0,:],xf1_ens[1,:],range=[[xmin,xmax],[ymin,ymax]], bins=50,density=True) 

plt.figure()
#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_etkf[0,:,0],x_ens_evol_etkf[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_etkf[0,:,1],x_ens_evol_etkf[1,:,1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_etkf[0,:,:]) , np.transpose( x_ens_evol_etkf[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x[0:-1],xf1_y[0:-1],np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Prior Den')
plt.contour(xa1_x[0:-1],xa1_y[0:-1],np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Posterior Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')

plt.legend()

plt.savefig('Figure_update_ETKF_1step_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')


#------------------------------------------------------------------------------
#  ETKF con TEMPERADO
#------------------------------------------------------------------------------

#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_etkft=np.zeros((nvars,EnsSize,NTemp+1)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_etkft[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.


tmp_for = xf1_ens_da 
Rt = R * NTemp
for itemp in range(NTemp) :
    [ tmp_for , tmp_anal_mean , Pa , null_var , null_var ] =da.analysis_update_ETKF( yo1 , tmp_for , forward_operator , Rt , 1.0 )

    x_ens_evol_etkft[:,:,itemp+1] = tmp_for

plt.figure()
#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_etkft[0,:,0],x_ens_evol_etkft[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_etkft[0,:,-1],x_ens_evol_etkft[1,:,-1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_etkft[0,:,:]) , np.transpose( x_ens_evol_etkft[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x[0:-1],xf1_y[0:-1],np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x[0:-1],xa1_y[0:-1],np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.plot(xt1[0],xt1[1],'ks',label='True')

plt.legend()

plt.savefig('Figure_update_ETKF_tempering_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')



#------------------------------------------------------------------------------
#  ETKF con RIP
#------------------------------------------------------------------------------

#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_etkfr=np.zeros((nvars,EnsSize,NRip+1)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_etkfr[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.
tmp_ens = np.zeros((nvars,EnsSize,2))
 
tmp_ens[:,:,0] = xa0_ens_da
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
plt.contour(xf1_x[0:-1],xf1_y[0:-1],np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.contour(xa1_x[0:-1],xa1_y[0:-1],np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05))
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.legend()

plt.savefig('Figure_update_ETKF_rip_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')


#------------------------------------------------------------------------------
#  ETPF en un paso
#------------------------------------------------------------------------------

#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
x_ens_evol_etpf=np.zeros((nvars,EnsSize,2)) #Solo 2 lugares, el muestreo del prior y el muestreo del posterior.

x_ens_evol_etpf[:,:,0] = xf1_ens_da #El tiempo 0 son las particulas sampleadas del prior.

tmp_for = xf1_ens_da 
[ tmp_anal , tmp_anal_mean , Pa , null_var , null_var ] =da.analysis_update_ETKF( yo1 , tmp_for , forward_operator , R , 1.0 )

x_ens_evol_etpf[:,:,1] = tmp_anal

plt.figure()
#plt.plot(yo1[0],yo1[1],'ks')
plt.plot(x_ens_evol_etpf[0,:,0],x_ens_evol_etpf[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_etpf[0,:,1],x_ens_evol_etpf[1,:,1],'bo',label='Posterior Ens')  #Posterior
plt.plot( np.transpose(x_ens_evol_etpf[0,:,:]) , np.transpose( x_ens_evol_etpf[1,:,:] ) , 'k--',linewidth=0.5)
plt.contour(xf1_x[0:-1],xf1_y[0:-1],np.transpose(xf1_den),colors='r',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Prior Den')
plt.contour(xa1_x[0:-1],xa1_y[0:-1],np.transpose(xa1_den),colors='b',linewidths=0.4,levels=np.arange(0.001,0.5,0.05),label='Posterior Den')
plt.plot(xt1[0],xt1[1],'ks',label='True')

plt.legend()

plt.savefig('Figure_update_ETPF_1step_logaritmic_' + str(bst) + '_' + str(numtrans) + '.png')







