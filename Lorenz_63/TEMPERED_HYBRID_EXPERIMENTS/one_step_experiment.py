#%%
#!/usr/bin/env python
# coding: utf-8

#En este experimento genero el prior y el posterior usando un ensamble de muchos miembros.
#Luego tomo un subsample pequenio y aplico diferentes estrategias para ver como aproximan al posterior.
import sys

sys.path.append("../")

import os
os.environ["MKL_NUM_THREADS"]="10"
os.environ["OPENBLAS_NUM_THREADS"]="10"
os.environ["OMP_NUM_THREADS"]="10"
os.environ["NUMEXPR_NUM_THREADS"]="10"

#Importamos todos los modulos que van a ser usados en esta notebook
from tqdm import tqdm
import numpy as np
import scipy as sc
import Lorenz_63 as model
import Lorenz_63_DA as da
import matplotlib.pyplot as plt
import pickle as pkl

#Seleccionar aqui el operador de las observaciones que se desea usar.
from Lorenz_63_ObsOperator import forward_operator_nonlinear    as forward_operator
from Lorenz_63_ObsOperator import forward_operator_nonlinear_tl as forward_operator_tl

# Configuracion del sistema del modelo y del sistema de asimilacion.
da_exp=dict()  #Este diccionario va a contener las variables importantes para nuestro experimento.


#------------------------------------------------------------
# Especificamos los parametros que usara el modelo
#------------------------------------------------------------

a      = 10.0      # standard L63 10.0 
r      = 28.0      # standard L63 28.0
b      = 8.0/3.0   # standard L63 8.0/3.0

p=np.array([a,r,b])

pim=np.array([a,r,b])

#------------------------------------------------------------
# Model and experienet setup
#------------------------------------------------------------

dt=0.01            # Paso de tiempo para la integracion del modelo de Lorenz
x0=np.array([ 8.0 , 0.0 , 30.0 ])      # Condiciones iniciales para el spin-up del nature run (no cambiar)
numtrans=600                           # Tiempo de spin-up para generar la condicion inicial.

#------------------------------------------------------------
# Configuracion del sistema de asimilacion
#------------------------------------------------------------

dx0 = np.array([ 5.0 , 5.0 , 5.0 ])       # Error inicial de la estimacion. 
R0=3.0                                    # Varianza del error de las observaciones.
bst=12                                    # Cantidad de pasos de tiempo que vamos a hacer entre el estado t0 y t1 (el momento donde asimilamos datos)
nvars=3                                   # Numero de variables en el modelo de Lorenz (no tocar)

EnsSize_prior=int(10e6)                          #Numero de miembros para el sampleo del prior y el posterior usando Bayes.
EnsSize=30                                       #Numero de miembros para la asimilacion.

#Obtengo el numero de observaciones (lo obtengo directamente del forward operator)
nobs=np.size(forward_operator(np.array([0,0,0])))

#Definimos una matriz de error de las observaciones
R=R0*np.identity(nobs)   #En esta formulacion asumimos que los errores 
                                                       #en diferentes observaciones son todos iguales y 
#Creamos un vector de bias para las observaciones.
obs_bias=np.zeros(nobs)            #que no estan correlacionados entre si.

#Matriz de covarianza que vamos a usar para generar las perturbaciones en t0.
#En t0 asumimos que la pdf es gaussiana con esta matriz de covarianza. En t1 vamos a tener
#Una pdf mas o menos gaussiana dependiendo de que tan grande sea bst.
P0=np.array([[0.6 , 0.2 , 0.0 ],[0.2 , 0.6 , 0.0 ],[0.0 , 0.0 , 1.0 ]])
P0sq = sc.linalg.sqrtm( P0 )

#Definimos una matriz Q para compensar los efectos no lineales y posibles errores de modelo.
Q=0.0 * np.identity(3)
#Definimos la inflacion multiplicativa
MultInf=1.00   


#%%
#------------------------------------------------------------
# Generamos el ensamble grande para samplear el prior y la posterior.
#------------------------------------------------------------

# Generamos la verdadera evolucion del sistema ("nature run")
# Integramos el modelo durante varios pasos de tiempo para que la solucion converja al atractor.

x=np.copy(x0)
for i in range(numtrans)  :
   x = model.forward_model( x , p , dt )
   
#En este valor vamos a centrar la Gaussiana en t0.   
x0 = np.copy(x)

#Inicializamos el ciclo desde la media "climatologica" del sistema. Es decir no tenemos informacion precisa
#de donde esta el sistema al tiempo inicial.

xa0_ens = np.random.randn( nvars , EnsSize_prior )

xa0_ens = np.dot( P0sq , xa0_ens )

for iv in range(nvars)  :
    xa0_ens[iv,:]=xa0_ens[iv,:] + x0[iv]
    
#  
    
#Integramos el ensamble para obtener el prior. 
xf1_ens=np.zeros( np.shape( xa0_ens ) )       
x=np.copy( xa0_ens )
for j in range(bst)  :
    print(j)
    x = model.forward_model_ens( x , p , dt )
xf1_ens = np.copy(x)     


#[xa0_den,xa0_x,xa0_y]=np.histogram2d(xa0_ens[0,:],xa0_ens[1,:], bins=50) 
#[xf1_den,xf1_x,xf1_y]=np.histogram2d(xf1_ens[0,:],xf1_ens[1,:], bins=50) 




#Save the ensembles
f=open('Ens_'+str(bst)+'.pkl','wb')
pkl.dump([xa0_ens,xf1_ens,wf1_ens],f)
f.close()   
            





    





