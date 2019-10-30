#%%
#!/usr/bin/env python
# coding: utf-8

# Inicializacion. Cargamos los modulos necesarios


#Importamos todos los modulos que van a ser usados en esta notebook

import numpy as np
import matplotlib.pyplot as plt

import Lorenz_63 as model
import Lorenz_63_DA as da

    
#------------------------------------------------------------
# Especificamos los parametros que usara el modelo
#------------------------------------------------------------

a      = 10.0      # standard L63 10.0 
r      = 28.0      # standard L63 28.0
b      = 8.0/3.0   # standard L63 8.0/3.0

p=np.array([a,r,b])


dt=0.01            # Paso de tiempo para la integracion del modelo de Lorenz
numstep=1000       # Cantidad de ciclos de asimilacion.
x0=np.array([ 8.0 , 0.0 , 30.0 ])      # Condiciones iniciales para el spin-up del nature run (no cambiar)
numtrans=600                           # Tiempo de spin-up para generar el nature run (no cambiar)

bst=8                                     # Cantidad de pasos de tiempo entre 2 asimilaciones.
forecast_length = 50                      # Plazo de pronostico (debe ser al menos 1)

EnsSize=1000                                #Numero de miembros en el ensamble.
nvars=3                                     #Cantidad de variables en el estado

pert_amp = 1.0 

ini_time = 100                              

#------------------------------------------------------------
# Generamos la simulacion "verdad"
#------------------------------------------------------------

# Generamos la verdadera evolucion del sistema ("nature run")
# Integramos el modelo durante varios pasos de tiempo para que la solucion converja al atractor.

x=np.copy(x0)
for i in range( numtrans )  :
   x = model.forward_model( x , p , dt )
    
# Integramos la simulacion verdad (Guardamos la salida cada 8 pasos de tiempo)
# El resultado es almacenado en un array de numpy "state" con dimension (numstep,3)

state=np.zeros(( numstep , nvars ))

for i  in range( numstep ) :
    for j in range( bst )      :
        x = model.forward_model( x , p , dt )
      
    state[i,:]=x
    
#------------------------------------------------------------
# Ensamble ....
#------------------------------------------------------------
    
#Generamos EnsSize perturbaciones y vamos a integrar el modelo a partir de 
#todos los estados perturbados. 

ensamble = np.zeros(( nvars , EnsSize , forecast_length )) 

#Generamos las perturbaciones iniciales

ensamble[:,:,0] = np.random.randn( nvars , EnsSize ) * pert_amp
for iens in range(EnsSize)  :
    ensamble[:,iens,0] = ensamble[:,iens,0] + state[ini_time,:]

#Integramos el modelo 
for iens in range(EnsSize)  :
   x=np.copy(ensamble[:,iens,0])
   
   for i  in range( 1 , forecast_length ) :
      for j in range( bst )      :
         x = model.forward_model( x , p , dt )
      
      ensamble[:,iens,i]=x


#Grafico las EnsSize trayectorias
      
#Graficamos el conjunto / ensamble para alguna variable en particular.      
plt.figure()

var=1
for iens in range(EnsSize) :
   plt.plot(ensamble[var,iens,:],'k-')
   plt.plot(state[ini_time:ini_time+forecast_length,var])
   plt.plot(np.mean(ensamble[var,:,:],0),'r-')
plt.plot()


#Graficamos la dispersion del ensamble como funci[on del tiempo]
var=0

plt.figure()

for iens in range(EnsSize) :
   plt.plot(np.std(ensamble[var,:,:],0),'r-')
plt.plot()
      
      

#Hacemos histogramas para alguna variable a diferentes tiempos
var=1

for it in (0,1,5,10,20,49) :
    
    plt.figure()    
    
    hist=np.histogram(ensamble[var,:,it], bins=15)
    
    plt.plot( 0.5*(hist[1][1:]+hist[1][0:-1]),hist[0])
    


#Calculamos las matrices de covarianza a diferentes tiempos

pert=np.zeros( np.shape(ensamble) )
mean=np.mean( ensamble , 1 )

for iens in range(EnsSize)  :
    for it in range(forecast_length) :
       pert[:,iens,it] = ensamble[:,iens,it] - mean[:,it]
    
    
for it in (0,1,5,10,20,49)  :
    print('La covarianza para el tiempo ',it,' es :')
    cov= np.dot( pert[:,:,it] , np.transpose(pert[:,:,it] ) )/(EnsSize-1)
    print( cov )
    
    
for it in (0,1,5,10,20,49)  :
    plt.figure()
    plt.plot(ensamble[0,:,it],ensamble[2,:,it],'ok')
    
    
#Calculamos el modelo tangente lineal
optim_time=1
x = np.copy( state[ini_time,:] )
for i in range( optim_time ) :
   
    L_total=np.identity(3)  #Inicializo el propagante del modelo tangente lineal.
    for j in range( bst )  :
       #Calculamos el propagante del modelo tangente lineal en toda la ventana.
       L = model.tl_model( x , p , dt ) 
       L_total = np.dot( L , L_total ) 
       
       
tmp = np.dot(np.transpose(L_total) , L_total )
#Calculamos los autovectores y autovalores de la matriz L' L para obtener los vectores singulares.
S, V = np.linalg.eig( tmp )
   
print('Los valores singulares son')
print(S)
print('Los vectores singulares son')
print('Primer vector')
print(V[:,0])
print('Segundo vector')
print(V[:,1])
print('Tercer vector')
print(V[:,2]) 
  

