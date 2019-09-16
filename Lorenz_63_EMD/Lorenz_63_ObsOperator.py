#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 16:21:37 2019

@author: juan
"""

import numpy as np

#Definimos el operador de las observaciones y su tangente lineal
#Notar que la definicion de ambos operadores tiene que ser consistente (uno debe ser 
# el tangente lineal del otro, de lo contrario el sistema de asimilacion no funcionara
# correctamente )

def forward_operator_full( state )  :
       #Operador de las observaciones full que observa las 3 variables directamente
    
       H=np.identity(3)
       obs = np.matmul(H,state)
    
       return obs

def forward_operator_full_tl( state )  :
    
       #Tangente lineal del operador de las observaciones full que observa las 3 variables directamente
    
       H=np.identity(3)
    
       return H