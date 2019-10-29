#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:48:36 2019
http://www.cmp.caltech.edu/~mcc/Chaos_Course/Lesson7/Lyapunov.pdf
"""
# Inicializacion. Cargamos los modulos necesarios

#from tqdm import tqdm       #Para generar la barra que indica el progreso del loop.
import numpy as np
import Lorenz_63 as model
import matplotlib.pyplot as plt

#Dadas las columnas de una matriz A esta funcion ortonormaliza las columnas de A
def orthonormalize(A) :
   n = len(A)
   r=np.zeros( np.shape(A)[1] )
   r[0] = np.sqrt(A[:,0].dot(A[:,0]))
   A[:, 0] = A[:, 0] / r[0]
   
   for i in range(1, n):
      Ai = A[:, i]
      for j in range(0, i):
          Aj = A[:, j]
          t = Ai.dot(Aj)
          Ai = Ai - t * Aj
      r[i] = np.sqrt(Ai.dot(Ai))
      A[:, i] = Ai /r[i]
    
   return A , r 


def sorteig(eigenValues,eigenVectors) :
    
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    return eigenValues , eigenVectors

# Configuracion del sistema del modelo y del sistema de asimilacion.
da_exp=dict()  #Este diccionario va a contener las variables importantes para nuestro experimento.
    
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
da_exp['numstep']=5000       # Cantidad de ciclos de asimilacion.
da_exp['x0']=np.array([ 8.0 , 0.0 , 30.0 ])      # Condiciones iniciales para el spin-up del nature run (no cambiar)
da_exp['numtrans']=600                           # Tiempo de spin-up para generar el nature run (no cambiar)

#------------------------------------------------------------
# Configuracion del sistema de asimilacion
#------------------------------------------------------------

da_exp['bst']=8                                     # Cantidad de pasos de tiempo entre 2 asimilaciones.
da_exp['nvars']=3                                   # Numero de variables en el modelo de Lorenz (no tocar)


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

#En este array vamos a guardar los vectores de Lyapunov para cada punto del atractor.
da_exp['BLV']=np.zeros((da_exp['nvars'],da_exp['nvars'],da_exp['numstep']))
#En este array vamos a guardar los exponentes locales de lyapunov (es decir las tasas de crecimiento
#local de los vectores de lyapunov, la tasa global o los exponentes globales seran dados por los 
#el promedio temporal sobre un tiempo largo del logaritmo de estas tasas)
da_exp['LE']=np.zeros((da_exp['nvars'],da_exp['numstep']))

#SV contendra los vectores singulares para un periodo de optimizacion de 8 pasos de tiempo
#para cada punto del atractor.
da_exp['SV']=np.zeros((da_exp['nvars'],da_exp['nvars'],da_exp['numstep']))
#S contiene las tasas de crecimiento (los valores singulares) asociados a las direcciones
#de maximo crecimiento en cada punto del atractor.
da_exp['S']=np.zeros((da_exp['nvars'],da_exp['numstep']))

#BV contiene 3 bred vectors calculados estimando la dinamica de las perturbaciones utilizando
#el modelo no lineal en lugar del tangente lineal. 
da_exp['BV']=np.zeros((da_exp['nvars'],da_exp['nvars'],da_exp['numstep']))
#BE contiene la tasa de amplificación local de los bred vectors en cada punto del atractor.
da_exp['BE']=np.zeros((da_exp['nvars'],da_exp['numstep']))


BLV=np.eye(3) #Inicializamos los backward Lyapunov vectors y bred vectors como las columnas de la identidad.
BV=np.eye(3)
da_exp['BLV'][:,:,0] = BLV
da_exp['BV'][:,:,0] = BLV

for i  in range( da_exp['numstep'] ) :
    L_total=np.identity(3)  #Inicializo el propagante del modelo tangente lineal.
    for ib in range( np.shape(BV)[1] ) :
       BV[:,ib] = BV[:,ib] + x
       
    for j in range( da_exp['bst'] )      :
        x = model.forward_model( x , da_exp['p'] , da_exp['dt'] )
        
        #Calculo los bred vectores.
        for ib in range( np.shape(BV)[1] ) :
            BV[:,ib] = model.forward_model( BV[:,ib] , da_exp['p'] , da_exp['dt'] ) 
        

        #Calculo el modelo tangente lineal para este paso de integracion.
        L = model.tl_model( x , da_exp['p'] , da_exp['dt'] )
        #Acumulo el modelo tangente lineal para encontrar el modelo lineal de los 8 pasos de tiempo.
        L_total = np.dot( L , L_total ) 
 
    BLV=np.dot( L_total , BLV )
    #Cada bst pasos de tiempo ortogonalizo la base de vectores de lyapunov. 
    [BLV,r]=orthonormalize(BLV)
    #Guardo los vectores de lyapunov obtenidos y la tasa local de su crecimiento.
    da_exp['BLV'][:,:,i] = BLV
    da_exp['LE'][:,i] = r

    #Hacemos el rescaling de los BV
    for ib in range( np.shape(BV)[1] ) :
       #Calculo las tasas de creimiento
       da_exp['BE'][ib,i] = np.sqrt( np.sum( (BV[:,ib]-x)**2 ) )
       #Rescaleo la perturbacion.
       BV[:,ib] = ( BV[:,ib] - x )/da_exp['BE'][ib,i]
    da_exp['BV'][:,:,i] = BV

    #Usando el modelo tangente lineal en los bst pasos de tiempo obtengo los vectores singulares y sus tasas de 
    #crecimiento asociadas. 
    [ eigValues , eigVectors ] = np.linalg.eig( np.dot(np.transpose(L_total) , L_total ) )
    [ da_exp['S'][:,i] , da_exp['SV'][:,:,i] ] = sorteig( eigValues , eigVectors )
        
    #Finalmente guardamos el estado. 
    da_exp['state'][i,:]=x
    
    
#State define la trayectoria no lineal a partir de la cual se linealiza el sistema 
#para obtener los vectores de lyapunov.
    
#Graficamos los exponentes de lyapunov
print('Los exponentes de lyapunov globales son')

print( np.mean( np.log( da_exp['LE'] ),  1) )

plt.figure()
plt.title('Exponentes locales de Lyapunov asociados al primer vector de Lyapunov')
plt.scatter(da_exp['state'][:,0],da_exp['state'][:,1],c=np.log(da_exp['LE'][0,:]),cmap='seismic',vmin=-1,vmax=1)
plt.colorbar()

plt.figure()
plt.title('Exponentes locales de Lyapunov asociados al segundo vector de Lyapunov')
plt.scatter(da_exp['state'][:,0],da_exp['state'][:,1],c=np.log(da_exp['LE'][1,:]),cmap='seismic',vmin=-1,vmax=1)
plt.colorbar()

plt.figure()
plt.title('Exponentes locales de Lyapunov asociados al tercer vector de Lyapunov')
plt.scatter(da_exp['state'][:,0],da_exp['state'][:,1],c=np.log(da_exp['LE'][2,:]),cmap='seismic',vmin=-1,vmax=1)
plt.colorbar()



#Graficamos las tasas de crecimiento de los singular vectors
plt.figure()
plt.title('Log de los valores singulares asociados al primer vector singular')
plt.scatter(da_exp['state'][:,0],da_exp['state'][:,1],c=np.log(da_exp['S'][0,:]),cmap='seismic',vmin=-1.2,vmax=1.2)
plt.colorbar()

plt.figure()
plt.title('Log de los valores singulares asociados al segundo vector singular')
plt.scatter(da_exp['state'][:,0],da_exp['state'][:,1],c=np.log(da_exp['S'][1,:]),cmap='seismic',vmin=-1.2,vmax=1.2)
plt.colorbar()

plt.figure()
plt.title('Log de los valores singulares asociados al tercer vector singular')
plt.scatter(da_exp['state'][:,0],da_exp['state'][:,1],c=np.log(da_exp['S'][2,:]),cmap='seismic',vmin=-1.2,vmax=1.2)
plt.colorbar()


#Graficamos las tasas de crecimiento de los bred vectors
    
plt.figure()
plt.title('Log de los valores singulares asociados al primer bred vector')
plt.scatter(da_exp['state'][:,0],da_exp['state'][:,1],c=np.log(da_exp['BE'][0,:]),cmap='seismic',vmin=-1.2,vmax=1.2)
plt.colorbar()

plt.figure()
plt.title('Log de los valores singulares asociados al segundo bred vector')
plt.scatter(da_exp['state'][:,0],da_exp['state'][:,1],c=np.log(da_exp['BE'][1,:]),cmap='seismic',vmin=-1.2,vmax=1.2)
plt.colorbar()

plt.figure()
plt.title('Log de los valores singulares asociados al tercer bred vector')
plt.scatter(da_exp['state'][:,0],da_exp['state'][:,1],c=np.log(da_exp['BE'][2,:]),cmap='seismic',vmin=-1.2,vmax=1.2)
plt.colorbar()


plt.figure()
plt.plot(np.log(da_exp['LE'][0,-100:-1]),'r-' )
plt.plot(np.log(da_exp['LE'][1,-100:-1]),'g-' )
plt.plot(np.log(da_exp['LE'][2,-100:-1]),'b-' )


plt.plot(np.log(da_exp['S'][0,-100:-1]),'r--' )
plt.plot(np.log(da_exp['S'][1,-100:-1]),'g--' )
plt.plot(np.log(da_exp['S'][2,-100:-1]),'b--' )

plt.plot(np.log(da_exp['BE'][0,-100:-1]),'r:' )





import plotly.graph_objects as go
from plotly.offline import plot
    
 
fig = go.Figure(data=[go.Scatter3d(
        x=da_exp['state'][:,0] ,
        y=da_exp['state'][:,1] ,            
        z=da_exp['state'][:,2] ,            
        mode='markers' ,
        marker=dict(
                size=5,
                color= da_exp['BE'][0,:] , 
                colorscale='Viridis',
                opacity=0.8)
                    
        )] )  
plot(fig)  



















