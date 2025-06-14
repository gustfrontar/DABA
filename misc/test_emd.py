# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from emd import emd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import ot 


#Probamos scomo funciona la subrutina emd para obtener los pesos que transforman una 
#muestra con una distribucion (prior) en otra muestra con la distribucion deseada (posterior)

obs=0.5

#Genero la muestra inicial con distribucion uniforme

nparticles=1000

#las filas de x corresponden a las realizaciones (particulas) las columnas a las diferentes variables.
x=np.zeros((nparticles,1))  #Tengo que definir todas las variables como arrays 2D para que funcione la subrutina.

x[:,0] = np.random.rand(nparticles)
x_w = np.ones((nparticles,1)) / nparticles

y=np.copy(x)          #y e y_w es lo que describe la target distribution.

y_w= np.exp( -0.5*np.power(x-obs,2)/0.05 )  #Elijo los pesos de forma tal que el posterior sea Gaussiano.
y_w= y_w / np.sum(y_w)          #Los pesos deben estar normalizados.


[distance , flow2 ] = emd( x , y  , Y_weights = y_w , return_flows= True) 

D = cdist(x,y,'euclidean')
flow=ot.emd(x_w[:,0],y_w[:,0],D,numItermax=100000,log=False)


#Return_flows es una flag que activa o no el output de la transformacion que permite 
#transformar la distribucion a priori en la distribucion a posteriori.
x_a = np.zeros( np.shape(x) )
x_a2 = np.zeros( np.shape(x) )


for j in range(0,nparticles)  :
    
    for i in range(0,nparticles) :
    
       x_a[j,0] = x_a[j,0] + x[i,0] * flow[j,i]
       x_a2[j,0] = x_a2[j,0] + x[i,0] * flow2[j,i]
x_a = nparticles * x_a
x_a2 = nparticles * x_a2
#x_t = np.dot( np.transpose(x[:,0], flow  ) * nparticles

[x_hist,bin_limits] = np.histogram( x , range=(0,1) , bins=20 , weights=x_w )
x_hist = x_hist / np.sum(x_hist)
[y_hist,bin_limits] = np.histogram( y , range=(0,1) , bins=20 , weights=y_w )
y_hist = y_hist / np.sum(y_hist)
[xa_hist,bin_limits] = np.histogram( x_a , range=(0,1) , bins=20 , weights=x_w )
xa_hist = xa_hist / np.sum(xa_hist)

[xa2_hist,bin_limits] = np.histogram( x_a2 , range=(0,1) , bins=20 , weights=x_w )
xa2_hist = xa2_hist / np.sum(xa2_hist)



plt.figure()

bin_center = 0.5*(bin_limits[1:]+bin_limits[0:-1])

plt.plot(bin_center,x_hist,'b-',label='prior')
plt.plot(bin_center,y_hist,'g-',label='posterior')
plt.plot(bin_center,xa_hist,'r-',label='posterior EMD')
plt.plot(bin_center,xa2_hist,'r--',label='posterior EMD2')
plt.figure()
plt.plot(x,y_w,'o')




