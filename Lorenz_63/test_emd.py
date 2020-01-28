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
    
#Esta funcion resuelve mediante un metodo iterativo el problema del transporte optimo
#con un parametro de regularizacion lambda. 
D = da.sinkhorn_ot( statefens , w , lam = lam , max_iter=10000 )
M = np.power( cdist(np.transpose(statefens),np.transpose(statefens),'euclidean') , 2 ) 
D4=np.transpose( ot.emd(np.ones(EnsSize)/EnsSize,w,M,numItermax=1.0e9,log=False) ) * EnsSize



print('Using the fortran routine')
D2=pf.sinkhorn_ot( ne=EnsSize , wi=w , wt=np.ones(EnsSize)/EnsSize , m=M , lambda_reg=lam , stop_threshold=1.0e-8 , max_iter=10000 )
D3=pf.sinkhorn_ot_robust( ne=EnsSize , wi=w , wt=np.ones(EnsSize)/EnsSize , m=M , lambda_reg=lam , stop_threshold=1.0e-8 , max_iter=10000 )
    
stateaens = np.matmul( statefens , D ) 

import matplotlib.pyplot as plt

plt.pcolor(D);plt.colorbar()
plt.title('D1')
plt.show()

plt.pcolor(D2);plt.colorbar()
plt.title('D2')
plt.show()

plt.pcolor(D3);plt.colorbar()
plt.title('D3')
plt.show()

plt.pcolor(D4);plt.colorbar()
plt.title('D4')
plt.show()