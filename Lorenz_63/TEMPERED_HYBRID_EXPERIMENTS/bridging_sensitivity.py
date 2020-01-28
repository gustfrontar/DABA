#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:47:53 2020

@author: jruiz
"""
import numpy as np
import os
import pickle

import sys

sys.path.append("../")



#Seleccionar aqui el operador de las observaciones que se desea usar.
from Lorenz_63_ObsOperator import forward_operator_nonlinear    as forward_operator
from Lorenz_63_ObsOperator import forward_operator_nonlinear_tl as forward_operator_tl
import copy

import tempered_hybrid_module as thm 

import multiprocessing as mp


max_proc=10

# Configuracion del sistema del modelo y del sistema de asimilacion.
da_exp=dict()  #Este diccionario va a contener las variables importantes para nuestro experimento.

da_exp['obs_operator_name'] = 'nonlinear' #CUIDADO, esto tiene que ser consistente con el import que figura mas arriba.

da_exp['random_seed']=10
#------------------------------------------------------------
# Especificamos los parametros que usara el modelo
#------------------------------------------------------------
a      = 10.0      # standard L63 10.0 
r      = 28.0      # standard L63 28.0
b      = 8.0/3.0   # standard L63 8.0/3.0

da_exp['p']=np.array([a,r,b])
da_exp['pim']=np.array([a,r,b])

#------------------------------------------------------------
# Model and experienet setup
#------------------------------------------------------------

da_exp['dt']=0.01            # Paso de tiempo para la integracion del modelo de Lorenz
da_exp['numstep']=10000
# Cantidad de ciclos de asimilacion.
da_exp['x0']=np.array([ 8.0 , 0.0 , 30.0 ])      # Condiciones iniciales para el spin-up del nature run (no cambiar)
da_exp['numtrans']=600                           # Tiempo de spin-up para generar el nature run (no cambiar)

#------------------------------------------------------------
# Configuracion del sistema de asimilacion
#------------------------------------------------------------

da_exp['dx0'] = np.array([ 5.0 , 5.0 , 5.0 ])       # Error inicial de la estimacion. 
da_exp['R0']=2.0                                    # Varianza del error de las observaciones.
da_exp['bst']=16                                    # Cantidad de pasos de tiempo entre 2 asimilaciones.
da_exp['forecast_length'] = 2                      # Plazo de pronostico (debe ser al menos 1)
da_exp['nvars']=3                                  # Numero de variables en el modelo de Lorenz (no tocar)

da_exp['EnsSize']=30                                 #Numero de miembros en el ensamble.

da_exp['rtps_alpha'] =  0.0   #Relaxation to prior spread (Whitaker y Hamill 2012) # 0.6 es un buen parametro (no se usa por el momento)
da_exp['rejuv_param'] = 0.0   #Parametro de rejuvenecimiento (Acevedo y Reich 2017) #0.4 es un buen parametro
da_exp['multinf']=1.0       #Inflacion multiplicativa (solo se aplica al ETKF, no al ETPF)

#Obtengo el numero de observaciones (lo obtengo directamente del forward operator)
da_exp['nobs']=np.size(forward_operator(np.array([0,0,0])))

#Definimos una matriz de error de las observaciones
da_exp['R']=da_exp['R0']*np.identity(da_exp['nobs'])   #En esta formulacion asumimos que los errores 
                                                       #en diferentes observaciones son todos iguales y 
#Creamos un vector de bias para las observaciones.
da_exp['obs_bias']=np.zeros(da_exp['nobs'])            #que no estan correlacionados entre si.

da_exp['P_from_file']=False                             #Si vamos a leer la matriz P de un archivo.
da_exp['P_to_file']=False                               #Si vamos a estimar y guardar la matriz P a partir de los pronosticos.

da_exp['P0']=10.0*np.array([[0.6 , 0.5 , 0.0 ],[0.5 , 0.6 , 0.0 ],[0.0 , 0.0 , 1.0 ]])
#P=None

#Definimos una matriz Q para compensar los efectos no lineales y posibles errores de modelo.
da_exp['Q']=0.4 * np.identity(3)

da_exp['forward_operator'] = forward_operator
da_exp['forward_operator_tl'] = forward_operator_tl


da_exp['ntemp']=1                                # Numero de temperados (1 recupera un ciclo de DA tradicional)
da_exp['bridge']=0.0                             # Coeficiente de combiancion entre ETPF y ETKF. 0-ETKF puro, 1-ETPF puro.


filename = './Sensitivity_to_bridging_R'+str(da_exp['R0'])+'_bst'+str(da_exp['bst'])+'_ntemp'+str(da_exp['ntemp'])+'_'+da_exp['obs_operator_name']+'.pkl'

#Run experiments in parallel. 

os.environ['OMP_NUM_THREADS']="1"

#Create a list of configurations. We will then iterate over these configurations to run experiments in parallel.

da_exp_list=list()   #Here we will store all possible configurations that we want to run.


for ibridge in range( 0 , 11 )  :

    da_exp['bridge']= ibridge / 10.0
    
    da_exp_list.append( copy.deepcopy( da_exp ) )  #Add this configuration to the configuration list


pool = mp.Pool( min( max_proc , len( da_exp_list ) ) )

results = pool.map( thm.da_cycle_tempered_hybrid , da_exp_list )

pool.close()

with open( filename , 'wb') as handle:
    pickle.dump( results , handle, protocol=pickle.HIGHEST_PROTOCOL)






















