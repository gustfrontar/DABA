#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""


import numpy as np

import matplotlib.pyplot as plt

ntimes=1000

std_error_obs = 0.1 

omega_sq=2.0

dt=0.01

estado=np.zeros((ntimes,2))

estado[0,:]=np.array([0.0,1.0])


M=np.array([[ -np.power(dt,2)*omega_sq + 1.0 , dt ],[ -dt*omega_sq , 1.0 ]])


for it in range(1,ntimes)  :
    
    estado[it,:] = np.dot(M,estado[it-1,:])
    
    
time = np.arange(0,ntimes*dt,dt)    
    
plt.figure()    
plt.plot( time , estado[:,0] )
plt.xlabel('X')
plt.ylabel('Tiempo')
plt.title('X en función del tiempo')
plt.savefig('ejemplo1.png')                #Guardo la figura en un archivo.



#Generar observaciones de la variable X agregando ruido a la secuencia verdadera.


error_obs = np.random.randn(ntimes) * std_error_obs 

plt.figure()
plt.plot( time , error_obs )
plt.xlabel('Error')
plt.ylabel('Tiempo')
plt.title('Error de la observacion en función del tiempo')
plt.savefig('ejemplo2.png')                #Guardo la figura en un archivo.



observacion = error_obs + estado[:,0]
plt.figure()
plt.plot( time , observacion , 'ok')
plt.plot( time , estado[:,0] , '-b')
plt.xlabel('Error')
plt.ylabel('Tiempo')
plt.title('Observación y X en función del tiempo')
plt.savefig('ejemplo3.png')                #Guardo la figura en un archivo.














    




    