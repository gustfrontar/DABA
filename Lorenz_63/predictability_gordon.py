#%%
#!/usr/bin/env python
# coding: utf-8

# Inicializacion. Cargamos los modulos necesarios


#Importamos todos los modulos que van a ser usados en esta notebook

import numpy as np
import matplotlib.pyplot as plt




numstep=1000         # Cantidad de ciclos de asimilacion.
x0=0               # Condiciones iniciales para el spin-up del nature run (no cambiar)

EnsSize=1000       #Numero de miembros en el ensamble.

pert_amp = 1.0 
ini_time = 100

forecast_length=100                              


def gordon(x_in,k):

    x_out = 0.5*x_in + 25*x_in / (1+x_in**2) + 8*np.cos(1.2*(k-1)) + np.random.randn() * np.sqrt(10.0)
    
    return x_out

#Genero una simulacion con el modelo.

state = np.zeros((numstep))

state[0] = x0

for i in range(1,numstep) :
    
    state[i] = gordon( state[i-1] , i )
    
plt.figure()
plt.plot(state)


#Genero un ensamble de estados iniciales

ensamble=np.zeros((EnsSize,forecast_length))

for iens in range(EnsSize) :
    ensamble[iens,0] = state[ini_time] + np.random.randn() * pert_amp
    for i in range(1,forecast_length) :
        ensamble[iens,i] = gordon( ensamble[iens,i-1] , i )
        
        
plt.figure()
for iens in range(EnsSize) :
    plt.plot(ensamble[iens,:],'k-')
plt.plot(np.mean(ensamble[:,:],0),'r')


plt.figure()

    
    



