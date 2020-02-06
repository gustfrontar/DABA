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
       obs=np.sum( state , 0 )
       
       return obs
       
def forward_operator_integral_tl( state ) :
   
       H=np.ones( (1 , np.size(state)) )
       return H
    
def forward_operator_nonlinear( state )  :
       local_state = np.copy(state)
       #Operador de las observaciones full que observa las 3 variables directamente

       #obs = np.power( state , 3 ) / 3000.0
       #obs = np.power( state + 20 , 2 ) / 100.0
       #state[0] = state[0] + 25.0
       #state[1] = state[1] + 25.0
       #state[2] = state[2] + 25.0
       local_state = local_state + np.array([18,23,-3])
       
       #print(np.shape(state
       local_state[ local_state < 0.0001 ] = 0.0001
       obs = np.log( local_state ) *10.0
    
       return obs
   
def forward_operator_nonlinear_ens( state )  :
       local_state = np.copy(state)
       #Operador de las observaciones full que observa las 3 variables directamente

       local_state[0,:] = local_state[0,:] + 18.0
       local_state[1,:] = local_state[1,:] + 23.0
       local_state[2,:] = local_state[2,:] -3.0
       
       local_state[ local_state < 0.0001 ] = 0.0001
       obs = np.log( local_state ) *10.0
    
       return obs   
   
def forward_operator_nonlinearsum_ens( state )  :
       local_state = np.copy(state)
       #Operador de las observaciones full que observa las 3 variables directamente
       #Operador similar al usado por Bunch y Godsill 2016
       tmp_state = np.copy(state)
       tmp_state[0,:]=tmp_state[0,:] + 11.0
       tmp_state[1,:]=tmp_state[1,:] + 1.0
       obs = np.sqrt( np.sum( np.power( tmp_state[0:2,:] , 2 ) , 0 ) )
    
       return obs     

def forward_operator_nonlinearsum( state )  :
       local_state = np.copy(state)
       #Operador de las observaciones full que observa las 3 variables directamente
       #Operador similar al usado por Bunch y Godsill 2016
       tmp_state = np.copy(state)
       tmp_state[0]=tmp_state[0] + 11.0
       tmp_state[1]=tmp_state[1] + 1.0      
       obs = np.sqrt( np.sum( np.power( tmp_state[0:2] , 2 ) , 0 ) )
    
       return obs
    

def forward_operator_nonlinear_tl( state )  :
    
       #Tangente lineal del operador de las observaciones full que observa las 3 variables directamente
    
       #H=np.diag( 3.0*np.power(state,2)/3000.0 )
       #H = np.diag( 2.0 * (state+20) / 100.0 )
       H = 10.0*np.diag(  1.0 / (state+30.0) )
    
       return H
    

    