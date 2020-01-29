#%%
#!/usr/bin/env python
# coding: utf-8

# Inicializacion. Cargamos los modulos necesarios


#Importamos todos los modulos que van a ser usados en esta notebook
from tqdm import tqdm

import numpy as np
import Lorenz_63 as model
import Lorenz_63_DA as da

import sys
sys.path.append("../Lorenz_96/data_assimilation/")
from da import common_pf as pf


#Seleccionar aqui el operador de las observaciones que se desea usar.
from Lorenz_63_ObsOperator import forward_operator_onlyx    as forward_operator
from Lorenz_63_ObsOperator import forward_operator_onlyx_tl as forward_operator_tl

np.random.seed(10)

#------------------------------------------------------------
# Especificamos los parametros que usara el modelo
#------------------------------------------------------------
a      = 10.0      # standard L63 10.0 
r      = 28.0      # standard L63 28.0
b      = 8.0/3.0   # standard L63 8.0/3.0

p=np.array([a,r,b])
dt=0.01            # Paso de tiempo para la integracion del modelo de Lorenz
x0=np.array([ 8.0 , 0.0 , 30.0 ])      # Condiciones iniciales para el spin-up del nature run (no cambiar)
numtrans=600                           # Tiempo de spin-up para generar el nature run (no cambiar)

#------------------------------------------------------------
# Configuracion del sistema de asimilacion
#------------------------------------------------------------
dx0 = 1.0*np.array([ 5.0 , 5.0 , 5.0 ])       # Error inicial de la estimacion. 
R0=8.0                                    # Varianza del error de las observaciones.
nvars=3
EnsSize=30                                 #Numero de miembros en el ensamble.

nobs=np.size(forward_operator(np.array([0,0,0])))

#Definimos una matriz de error de las observaciones
R=R0*np.identity(nobs)   #En esta formulacion asumimos que los errores 
                                                       #en diferentes observaciones son todos iguales y 
P0=10.0*np.array([[0.6 , 0.5 , 0.0 ],[0.5 , 0.6 , 0.0 ],[0.0 , 0.0 , 1.0 ]])

lam = 40.0

x=np.copy(x0)
for i in range(numtrans)  :
   x = model.forward_model( x , p , dt )
    
# Integramos la simulacion verdad
# El resultado es almacenado en un array de numpy "state" con dimension (numstep,3)

yo = forward_operator( x ) + np.random.multivariate_normal(np.array([0]),R)

#Inicializamos el ciclo desde la media "climatologica" del sistema. Es decir no tenemos informacion precisa
#de donde esta el sistema al tiempo inicial.

statefens=np.zeros((nvars,EnsSize))

for iens in range( EnsSize ) :
    statefens[:,iens] = np.nanmean( x , 0 ) + dx0 + np.random.multivariate_normal(np.zeros(nvars),P0)
    

#Calculamos la matriz de transporte opitmo.
#from emd import emd
from scipy.spatial.distance import cdist
import ot
 
    
    
#Calculo la inversa de la matriz de covarianza    
Rinv = np.linalg.inv(R)
        
#Calculamos los pesos en base al likelihood de las observaciones 
#dada cada una de las particulas.
w=np.zeros( EnsSize )
for iens in range( EnsSize ) :
    yf = forward_operator( statefens[:,iens] )
    w[iens] = np.exp( -0.5 * np.matmul( (yo-yf).transpose() , np.matmul( Rinv , yo - yf ) ) )

#Normalizamos los pesos para que sumen 1.
w = w / np.sum(w)
  

  
#Compute the weights as in the fortran routine
w2=np.zeros(EnsSize)
for iens in range(EnsSize) :
   yf = forward_operator( statefens[:,iens] )
   w2[iens]=w2[iens] - 0.5*( (yo-yf)**2 ) * Rinv 

  
 #Normalize log of the weigths (to avoid underflow issues)
log_w_sum = da.log_sum_vec(w2)

for iens in range(EnsSize) :

   w2[iens] = np.exp( w2[iens] - log_w_sum )
   #w2[iens] = np.exp(w2[iens])

#Normalizamos los pesos para que sumen 1.
w2 = w2 / np.sum(w2)

import matplotlib.pyplot as plt
plt.plot(w2);plt.plot(w)






