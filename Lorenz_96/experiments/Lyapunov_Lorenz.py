#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:48:36 2019

@author: jruiz
"""
# Inicializacion. Cargamos los modulos necesarios


#Importamos todos los modulos que van a ser usados en esta notebook

#from tqdm import tqdm       #Para generar la barra que indica el progreso del loop.
import numpy as np
import Lorenz_63 as model
import Lorenz_63_DA as da

#Seleccionar aqui el operador de las observaciones que se desea usar.
from Lorenz_63_ObsOperator import forward_operator_full    as forward_operator
from Lorenz_63_ObsOperator import forward_operator_full_tl as forward_operator_tl

# Configuracion del sistema del modelo y del sistema de asimilacion.
da_exp=dict()  #Este diccionario va a contener las variables importantes para nuestro experimento.

da_exp['exp_id']='3DV_xyz'    #Este es un identificador que se agregara al nombre de los archivos generados por este script (figuras, pickle contiendo los datos, etc)
da_exp['main_path']='./' + da_exp['exp_id']


#%%

#----------------------------------------------------------------------
# Creamos los directorios donde se guardaran los datos del experimento
#----------------------------------------------------------------------
    
da.directory_init( da_exp )
    
#------------------------------------------------------------
# Especificamos los parametros que usara el modelo
#------------------------------------------------------------

a      = 10.0      # standard L63 10.0 
r      = 28.0      # standard L63 28.0
b      = 8.0/3.0   # standard L63 8.0/3.0

da_exp['p']=np.array([a,r,b])

#------------------------------------------------------------
# Model and experienet setup
#------------------------------------------------------------

da_exp['dt']=0.01            # Paso de tiempo para la integracion del modelo de Lorenz
da_exp['numstep']=1000       # Cantidad de ciclos de asimilacion.
da_exp['x0']=np.array([ 8.0 , 0.0 , 30.0 ])      # Condiciones iniciales para el spin-up del nature run (no cambiar)
da_exp['numtrans']=600                           # Tiempo de spin-up para generar el nature run (no cambiar)

#------------------------------------------------------------
# Configuracion del sistema de asimilacion
#------------------------------------------------------------

da_exp['R0']=2.0                                    # Varianza del error de las observaciones.
da_exp['bst']=8                                     # Cantidad de pasos de tiempo entre 2 asimilaciones.
da_exp['forecast_length'] = 20                      # Plazo de pronostico (debe ser al menos 1)
da_exp['nvars']=3                                   # Numero de variables en el modelo de Lorenz (no tocar)

#Obtengo el numero de observaciones (lo obtengo directamente del forward operator)
da_exp['nobs']=np.shape(forward_operator(np.array([0,0,0])))[0]

#Definimos una matriz de error de las observaciones
da_exp['R']=da_exp['R0']*np.identity(da_exp['nobs'])   #En esta formulacion asumimos que los errores 
                                                       #en diferentes observaciones son todos iguales y 
#Creamos un vector de bias para las observaciones.
da_exp['obs_bias']=np.zeros(da_exp['nobs'])            #que no estan correlacionados entre si.

da_exp['P_from_file']=False                             #Si vamos a leer la matriz P de un archivo.
da_exp['P_to_file']=True                               #Si vamos a estimar y guardar la matriz P a partir de los pronosticos.

P=np.array([[0.6 , 0.5 , 0.0 ],[0.5 , 0.6 , 0.0 ],[0.0 , 0.0 , 1.0 ]])
#P=None


#%%
#------------------------------------------------------------
# Generamos la simulacion "verdad"
#------------------------------------------------------------

# Generamos la verdadera evolucion del sistema ("nature run")
# Integramos el modelo durante varios pasos de tiempo para que la solucion converja al atractor.

x=np.copy(da_exp['x0'])
for i in range(da_exp['numtrans'])  :
   x = model.forward_model( x , da_exp['p'] , da_exp['dt'] )
    
# Integramos la simulacion verdad
# El resultado es almacenado en un array de numpy "state" con dimension (numstep,3)

da_exp['state']=np.zeros((da_exp['numstep'],da_exp['nvars']))

for i  in range( da_exp['numstep'] ) :
    for j in range( da_exp['bst'] )      :
        x = model.forward_model( x , da_exp['p'] , da_exp['dt'] )
      
    da_exp['state'][i,:]=x
    
    
    


function lorenz_spectra(T,dt)
% Usage: lorenz_spectra(T,dt)
% T is the total time and dt is the time step
% parameters defining canonical Lorenz attractor
sig=10.0;
rho=28;
bet=8/3;
% dt=0.01; %time step
N=T/dt; %number of time intervals
% calculate orbit at regular time steps on [0,T]
% using matlab’s built-in ode45 runke kutta integration routine
% begin with initial conditions (1,2,3)
x1=1; x2=2; x3=3;
% integrate forwards 10 units
[t,x] = ode45(’g’,[0:1:10],[x1;x2;x3]);
n=length(t);
% begin at this point, hopefully near attractor!
x1=x(n,1); x2=x(n,2); x3=x(n,3);
[t,x] = ode45(’g’,[0:dt:T],[x1;x2;x3]);
e1=0;
e2=0;
e3=0;
% show trajectory being analyzed
plot3(x(:,1),x(:,2),x(:,3),’.’,’MarkerSize’,2);
JN = eye(3);
w = eye(3);
J = eye(3);
for k=1:N
% calculate next point on trajectory
x1 = x(k,1);
x2 = x(k,2);
x3 = x(k,3);
% calculate value of flow matrix at orbital point
% remember it is I+Df(v0)*dt not Df(v0)
J = (eye(3)+[-sig,sig,0;-x3+rho,-1,-x1;x2,x1,-bet]*dt);
% calculate image of unit ball under J
% remember, w is orthonormal ...
3
w = ortho(J*w);
% calculate stretching
% should be e1=e1+log(norm(w(:,1)))/dt; but scale after summing
e1=e1+log(norm(w(:,1)));
e2=e2+log(norm(w(:,2)));
e3=e3+log(norm(w(:,3)));
% e1=e1+norm(w(:,1))-1;
% e2=e2+norm(w(:,2))-1;
% e3=e3+norm(w(:,3))-1;
% renormalize into orthogonal vectors
w(:,1) = w(:,1)/norm(w(:,1));
w(:,2) = w(:,2)/norm(w(:,2));
w(:,3) = w(:,3)/norm(w(:,3));
end
% exponent is given as average e1/(N*dt)=e1/T
e1=e1/T; % Lyapunov exponents
e2=e2/T;
e3=e3/T;
l1=exp(e1); % Lyapunov numbers
l2=exp(e2);
l3=exp(e3);
[e1,e2,e3]
trace=e1+e2+e3
[l1,l2,l3]
The output is given by
>> lorenz_spectra(10,0.00