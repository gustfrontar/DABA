#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:32:18 2020

@author: jruiz


"""
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

import sys

sys.path.append("../")

#Es necesario importar estos modulos porque la funcion guardada en los resultados refiere a ellos.
#Esto sugiere que la funcion no se guarda, sino que se guarda su referencia.
#Es decir que si se cambia el modulo tambien se cambia la funcion.
import Lorenz_63_ObsOperator  
import Lorenz_63_ObsOperator 


results = pickle.load( open('Sensitivity_to_temperingiterations_R2.0_bst16_alpha0.0_nonlinear.pkl', "rb" ) )

NExp = len(results)

RmseA = np.zeros(NExp)
SprdA = np.zeros(NExp)


for iexp in range(NExp)  :
    
    RmseA[iexp]=np.sum(results[iexp]['rmse_a'])
    SprdA[iexp]=np.sum( np.mean( results[iexp]['stateasprd'] , 0 ) )
    
    