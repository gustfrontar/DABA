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
   
def forward_operator_onlyx( state )  :
       #Operador de las observaciones full que observa las 3 variables directamente
    
       obs = np.array(state[0])
    
       return obs

def forward_operator_onlyx_tl( state )  :
    
       #Tangente lineal del operador de las observaciones full que observa las 3 variables directamente
    
       H=np.array([1,0,0])
    
       return H

    
   
def forward_operator_integral( state ) :
       obs=np.sum( state )
       
       return obs
       
def forward_operator_integral_tl( state ) :
   
       H=np.ones( (1 , np.size(state)) )
       return H
    
def forward_operator_nonlinear( state )  :
       #Operador de las observaciones full que observa las 3 variables directamente

       obs = np.power( state , 3 ) / 1000.0
    
       return obs

def forward_operator_nonlinear_tl( state )  :
    
       #Tangente lineal del operador de las observaciones full que observa las 3 variables directamente
    
       H=np.diag( 3.0*np.power(state,2)/1000.0 )
    
       return H

    