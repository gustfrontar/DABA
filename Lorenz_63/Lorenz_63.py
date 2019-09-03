# -*- coding: utf-8 -*-
"""
Este modulo contiene las funciones necesarias para integrar el modelo de Lorenz (Lorenz 63)
y su tangente lineal.
"""

import numpy as np

def forward_model( xi , p , dt )  :

#xnl es la condicion inicial total (para la trayectoria no lineal)
#xi es la perturbacion 
  
#Calculo la trayectoria no lineal dentro del paso de tiempo alrededor de la
#cual voy a obtener el tangente lineal y el adjunto.

   f = l63eq( xi , p )
   
   c1 = dt * f  
   xa = xi + c1 / 2.0
   f = l63eq( xa , p )
   
   c2 = dt * f  
   xb = xi + c2 / 2.0 
   f = l63eq( xb , p )
   
   c3 = dt * f  
   xc = xi + c3  
   f = l63eq( xc , p )
   
   c4 = dt * f  
   xend = xi + (c1 + 2.0*c2 + 2.0*c3 + c4) / 6.0
   
   return xend


def tl_model( xi , p , dt ) :

#xi es la condicion inicial total (para la trayectoria no lineal)
#L es el propagante del tangente lineal y L_ad es su adjunto.
   
#Calculo la trayectoria no lineal dentro del paso de tiempo alrededor de la
#cual voy a obtener el tangente lineal y el adjunto.

   f = l63eq( xi , p )
   
   c1 = dt * f   
   xa = xi + c1 / 2.0
   f = l63eq( xa , p )
   
   c2 = dt * f   
   xb = xi + c2 / 2.0 
   f = l63eq( xb , p )
   
   c3 = dt * f  
   xc = xi + c3  
   
   #Vamos a calcular el modelo tangente lineal.
   
   I=np.identity(3)
   Ji=l63eq_tl( xi , p )
   Ja=l63eq_tl( xa , p )
   Jb=l63eq_tl( xb , p )
   Jc=l63eq_tl( xc , p )
   
   #L = I + dt * ( 1.0 / 6.0 ) * ( Ji + 2.0 * Ja * ( Ji * dt / 2.0 + I ) + 2.0 * Jb * ( Ja * ( Ji * dt / 2.0 + I ) * dt / 2.0 + I )
   #     + Jc * ( dt * Jb * ( Ja * ( Ji * dt / 2.0 + I ) * dt / 2.0 + I ) + I ) )
  
   J1=Ji + 2.0 * np.dot( Ja , ( Ji * dt / 2.0 + I ) )
   J2=2.0 * np.dot( Jb , ( np.dot( Ja , ( Ji * dt / 2.0 + I ) ) * dt / 2.0 + I ) ) 
   J3=np.dot( Jc , dt * np.dot( Jb , ( np.dot( Ja , ( Ji * dt / 2.0 + I ) ) * dt / 2.0 + I ) ) + I ) + I 
   
   L = I + dt * ( 1.0 / 6.0 ) * ( J1 + J2 + J3 )
   
   #Devuelve el propagante del tangente lineal (el script de matlab devolvia su adjunto)
    
   return L 


def  l63eq( x , par ) :
   #Ecuaciones del modelo de Lorenz.
   #Dado un x esta funcion devuelve la derivada de x respecto del tiempo.
   #x es un vector de 3 elementos donde cada elemento representa una de las variables
   #del modelo de Lorenz
    
   x_dot = np.zeros(3)
    
   a      = par[0]
   r      = par[1]
   b      = par[2]
 
   x_dot[0] = a*(x[1] - x[0])
   x_dot[1] = -x[0]*x[2] + r*x[0] - x[1]
   x_dot[2] = x[0]*x[1] - b*x[2]

   return x_dot

def l63eq_tl( x , par )  :

   #Tangente lineal de las ecuaciones de Lorenz.
   #x es el estado de la trayectoria no lineal alrededor de la cual se
   #calcula el modelo tangente lineal.

   M=np.zeros((3,3)) #Tangent linear model (M)

   a      = par[0]
   r      = par[1]
   b      = par[2]
   
   M=np.zeros((3,3))
   M[0,0]=-a
   M[0,1]=a
   M[0,2]=0
   M[1,0]=-x[2] + r
   M[1,1]=-1
   M[1,2]=-x[0]
   M[2,0]=x[1] 
   M[2,1]=x[0]
   M[2,2]=-b   

   #M= np.matrix([[-a , a , 0],[-x[2] + r , -1 , -x[0]],[x[1] , x[0] , -b]])

   return M 




