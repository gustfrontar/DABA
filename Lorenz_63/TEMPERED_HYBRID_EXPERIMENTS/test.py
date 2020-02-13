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
from Lorenz_63_ObsOperator import forward_operator_full       as forward_operator
from Lorenz_63_ObsOperator import forward_operator_full_tl     as forward_operator_tl

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

R0=1.0
R=R0*np.eye(nobs)


np.random.seed(20)  #Fix the random seed.


f=open('Ens_'+str(bst)+ '_' + str(numtrans) + '.pkl','rb')
[xa0_ens,xf1_ens] = pkl.load(f)
f.close()  

SampleSize = xa0_ens.shape[1]


#First chose a random member which will be the true. We can chose the last one insted of a random one.

xt0 = xa0_ens[:,-1]
xt1 = xf1_ens[:,-1]    #No model error

#Second generate an observation from the true
yo1 = forward_operator( xt1 ) + np.random.multivariate_normal(np.zeros(nobs),R*np.eye(nobs))
 
#Third compute the full posterior density using importance sampling
#w=np.zeros(xa0_ens.shape[1])
#yf1_ens = forward_operator_ens( xf1_ens )
#dep = np.tile(yo1,(SampleSize,1)).transpose() - yf1_ens
#w=  np.exp( -np.sum( np.power( dep , 2 ) , 0 ) / (2.0*R0) )  #This is a simplified version of the computation of the likelihood.
#w = w / np.sum(w) #Weigth normalization.

#Third generate an ensemble of states that will represent the ensemble available for data assimilation.
#Podemos elegirlas de manera aleatoria o ir por las primeras EnsSize y ya... (voy por Homero Simpson y ya)

xa0_ens_da = xa0_ens[:,0:EnsSize]
xf1_ens_da = xf1_ens[:,0:EnsSize]


#Ahora vamos a definir diferentes variantes de asimilacion y vamos a guardar la evolucion
#entre la forma en la que aproximan el prior y la forma en la que aproximan el posterior y 
#todos los pasos intermedios si los ubiera.

#------------------------------------------------------------------------------
#  GPF en un solo paso (Este metodo es esencialmente un temperado con lo cual
#  no tiene sentido probar temperado para este metodo pero si quizas RIP)
#------------------------------------------------------------------------------

#Esta variable va a guardar la evolucion de las particulas durante la asimilacion.
import Lorenz_63_DA as da
tmp_for = np.copy( xf1_ens_da ) 
[ tmp_anal , tmp_anal_mean , x_ens_evol_gpf ] = da.analysis_update_GPF( yo1 , tmp_for , forward_operator , forward_operator_tl , R )


plt.figure()
plt.plot(x_ens_evol_gpf[0,:,0],x_ens_evol_gpf[1,:,0],'ro',label='Prior Ens')  #Prior
plt.plot(x_ens_evol_gpf[0,:,-1],x_ens_evol_gpf[1,:,-1],'bo',label='Posterior Ens',markersize=4.0)  #Posterior
plt.plot( np.transpose(x_ens_evol_gpf[0,:,:]) , np.transpose( x_ens_evol_gpf[1,:,:] ) , 'k--',linewidth=0.5)
plt.plot(xt1[0],xt1[1],'ks',label='True')
plt.show()
#plt.legend()
print('Pepe')
#plt.savefig('Figure_update_GPF_1step_linear_' + str(bst) + '_' + str(numtrans) + '.png')

#Ploteo la evolucion de las particulas en cada dimension en el pseudo tiempo.
#Ntimes = x_ens_evol_gpf.shape[2]
#pseudo_time = np.arange( 0 , Ntimes , 1 ) / ( Ntimes - 1 )
#pseudo_time_mat = np.tile(pseudo_time,(EnsSize,1)).transpose()

#plt.figure(figsize=(10,4))
#plt.plot(yo1[0],yo1[1],'ks')
#plt.subplot(1,3,1)
#plt.plot(x_ens_evol_gpf[0,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
#plt.plot(x_ens_evol_gpf[0,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
#plt.plot(x_ens_evol_gpf[0,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
#plt.plot(xf1_x,xf1_xden/2.0,color='r',linewidth=0.4,label='Prior Den')
#plt.plot(xa1_x,xa1_xden/2.0,color='b',linewidth=0.4,label='Posterior Den')
#plt.title('X')
#plt.plot(xt1[0],1,'ks',label='True')
#plt.subplot(1,3,2)
#plt.plot(x_ens_evol_gpf[1,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
#plt.plot(x_ens_evol_gpf[1,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
#plt.plot(x_ens_evol_gpf[1,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
#plt.plot(xf1_y,xf1_yden/2.0,color='r',linewidth=0.4,label='Prior Den')
#plt.plot(xa1_y,xa1_yden/2.0,color='b',linewidth=0.4,label='Posterior Den')
#plt.plot(xt1[1],1,'ks',label='True')
#plt.title('Y')
#plt.subplot(1,3,3)
#plt.plot(x_ens_evol_gpf[2,:,0],np.zeros(EnsSize),'ro',markersize=3.0,label='Prior Ens')      #Prior
#plt.plot(x_ens_evol_gpf[2,:,-1],np.ones(EnsSize),'bo',markersize=3.0,label='Posterior Ens')  #Posterior
#plt.plot(x_ens_evol_gpf[2,:,:].transpose(),pseudo_time_mat, 'k--',linewidth=0.5)             #Evolution
#plt.plot(xf1_z,xf1_zden/2.0,color='r',linewidth=0.4,label='Prior Den')
#plt.plot(xa1_z,xa1_zden/2.0,color='b',linewidth=0.4,label='Posterior Den')
#plt.plot(xt1[2],1,'ks',label='True')
#plt.title('Z')
#plt.legend()
#plt.savefig('Figure_marginal_dist_GPF_1step_linear_' + str(bst) + '_' + str(numtrans) + '.png')




