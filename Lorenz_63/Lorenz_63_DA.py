#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Este modulo contiene una serie de funciones para aplicar diferentes metodos de 
asimilacion de datos utilizando el modelo de Lorenz de 3 dimensiones (Lorenz63)

"""
import numpy as np
import scipy.linalg as linalg 


def gen_obs( h , da_exp )  :

#  Esta funcion genera las observaciones y genera las matrices necesarias para 
#  realizar la asimilacion.
    
#  input : 
#  numstep - number of analysis cycle
#  h -   Forward operator que transforma de las variables del modelo a las variables observadas
#  R0  - Varianza del error de las observaciones (asummos que todas las obs tienen el mismo error)
#  state - Array conteniendo el valor de la "verdadera" evalucion del sistema para todos los tiempos.
#    
#  Shu-Chih Yang 2005
#  Adaptado a python DABA 2019  

   #Obtengo la cantidad de tiempos a partir de la cual vamos a generar las observaciones.
   ntimes = da_exp['state'].shape[0]
    
   #Generamos el array donde se gardaran las observaciones
   da_exp['yobs'] = np.zeros( ( ntimes , da_exp['nobs'] ) )
   
   #Generamos las observaciones, tomamos los valores del true y le agregamos un ruido
   #Gaussiano para simular el error de la observacion.
   
   for it in range(ntimes)  :
       error=np.random.multivariate_normal(da_exp['obs_bias'],da_exp['R'])
       da_exp['yobs'][it,:] = h(da_exp['state'][it,:]) + error
       
   return  da_exp


def get_P( da_exp , P=None ) :
    
    
   #Definimos la matriz de covarianza de los errores del campo preliminar. 
   #Tres posibilidades:
   #a) Que P sea ingresada como parametro (opcional) a esta funcion. En ese caso la matriz de covarianza P
   #toma la matriz que ingresamos.
   #b) Se lee la matriz P de un archivo (este archivo se genera al haber corrido el filtro anteriormente con la opcion P_to_file = True)
   #c) Se usa la climatologia y se estima la matriz P como alpha * P' donde P' es la matriz de covarianza de las variables del estado
   #se asume que el error del analisis es un porcentaje pequeño de esa variabilidad. 
   
   fail=False
   
   
   if P is None :    #No recibimos una matriz P de entrada.
 
      if da_exp['P_from_file']  :
          file = da_exp['main_path'] + '/data/P_OI.pkl'# #En caso que sea true de que archivo?
          try  :
              import pickle 
              nfile= open( file , 'rb' )
              da_exp['P0'] = pickle.load( nfile ) 
          except :
              print('WARNING: No pude leer la matriz P del archivo: ' + file )
              fail=True
      if not da_exp['P_from_file'] or fail  :
         print('Estimamos P a partir del nature run')
         da_exp['P0'] = 0.01  * np.matmul( da_exp['state'].transpose() , da_exp['state'] ) / ( da_exp['numstep'] -1 ) 
   else  :
       print('Vamos a usar una P definida por el usuario')
       da_exp['P0'] = P 
       
       
   print('La matriz que vamos a usar es:')
   print(da_exp['P0'])
    
   return da_exp


def estimate_P( da_exp ) :
    
    if da_exp['P_to_file'] :
    
       import pickle 
    
       #Esta funcion usa los pronosticos para estimar la matriz de covarianza.
       #La idea es que el pronostico que la diferencia entre dos pronosticos que verifican al mismo tiempo
       #pero que fueron inicializados en 2 tiempos diferentes pueden constituyen un proxy del error del pronostico
       #y que la estadistica de estas diferencias aproxima la estadistica de los errorers (varianzas y covarianzas)
       alfa = 0.5 #Parametro que permite ajustar la estimacion de P.
    
       spin_up = 200  #Removemos los primeros 200 ciclos de asimilacion.
       #Para eso tomamos el array de pronosticos.
       if spin_up > da_exp['numstep'] :
          print('WARNING: No hay suficientes tiempos en el analisis para estimar P')
          return
       if da_exp['forecast_length'] <= 2 :
          print('WARNING: No tengo suficientes plazos de pronostico para estimar P')
          return
    
    
       statef = da_exp['statef'][spin_up:,:,:]
       #Calculamos la diferencia entre pronosticos inicializados a diferentes tiempos que 
       #verifican a la misma hora.
       forecast_diff = statef[:,:,2] - statef[:,:,1]  
       ntimes = np.shape(forecast_diff)[0]

       P_est = alfa  * np.matmul( forecast_diff.transpose() , forecast_diff ) / ( ntimes -1 )
    
       #Guardamos la estimacion en un archivo
       file = da_exp['main_path'] + '/data/P_OI.pkl'# #En caso que sea true de que archivo?
       nfile= open( file , 'wb' )
       pickle.dump( P_est , nfile ) 
       
       print('La matriz P que estime es:')
       print(P_est)

    

#FUNCIONES PARA EL PASO DE ASIMILACION SEGUN DIFERENTES METODOS.
   
def analysis_update( yo , xf , P , forward_operator , forward_operator_tl , R )   :

   #---------------------------------------------
   #  Dada matrices R, P, un campo preliminar y un conjunto de observaciones
   #  calcula el analisis y la matriz de covarianza de los errores del analisis
   #
   #  Los metodos que usan esta funcion son el OI / EKF
   #
   #  input:
   #       yo      - observaciones
   #       xf(dim) - campo preliminar (first gues)
   #       P       - matriz de covarianza de los errores del campo preliminar
   #       R       - matriz de covarianza de los errores de las observaciones
   #       forward_operator - funcion que pasa del espacio del estado al espacio de las observaciones
   #       forward_operator_tl - tangente lineal del forward opeator
   #  output:
   #       xa(dim) - analysis
   #       Pa      - matriz de covarianza de los errores del analisis.
   #
   #---------------------------------------------

   H=forward_operator_tl( xf )

   #Calculo la ganancia de Kalman. 
   # K = PH^t * inv( H P H^t + R )
   
   PHt=np.matmul( P  , H.transpose() )
   HPHt=np.matmul( H , PHt )
   HPHtinv=np.linalg.inv( HPHt + R )
   
   K = np.matmul( PHt , HPHtinv )
   #Aplicamos el forward operator
   Hxf=forward_operator( xf )
   
   #Calculo el incremento del analisis y el analisis propiamente dicho
   xa = xf + np.matmul(K , yo - Hxf )

   #Calculo la matriz identidad con la misma dimension que la matriz de covarianza
   #de los errores del campo preliminar.
   
   I = np.identity( P.shape[0] )
   
   #Calculo la matriz de covarianza de los errores del analisis
   ImKH=(I - np.matmul(K,H) )
   Pa = np.matmul( ImKH, P )
   
   OmB=yo-Hxf                         #To compute O-B statistics
   OmA=yo - forward_operator( xa )    #To compute A-B statistics
   
   return xa , Pa , OmB , OmA 


def analysis_update_POEnKF( yo , xf , forward_operator , forward_operator_tl , R , Inflation )   :

   #---------------------------------------------#   
   #  Dada matrices R, un ensamble de campos preliminares y un conjunto de observaciones
   #  calcula el analisis, la matriz de covarianza de los errores del analisis
   #  y las perturbaciones para integrar el ensamble durante el tiempo siguiente.
   #
   #  Los metodos que usan esta funcion son el EnKF con observaciones perturbadas
   #  Burguers et al 1998
   #
   #
   #  input:
   #       yo      - observaciones
   #       xf(dim,nens) - campo preliminar (first gues)  ensemble
   #       forward_operator - operador de las observaciones (funcion)
   #       forward_operator_tl - tangente lineal del operador de las observaciones (funcion)
   #  output:
   #       xa(dim,nens) - analysis ensemble
   #
   #---------------------------------------------

   #En esta formulacion aplicamos el update a cada miembro del ensamble. Pero 
   #para garantizar la consistencia entre el ensamble de analisis y la Pa que 
   #predice la teoria de Kalman, las observaciones deben ser perturbadas.
   #El update es calculado para cada miembro con la misma K pero con observaciones
   #ligeramente diferentes.
   
    #Obtengo la cantidad de miembros en el ensamble
    [nvar , nens ]= xf.shape
   
    #Dado que el estado es pequeño puedo calcular Pf explicitamente

    #Defino la matriz que guardara las perturbaciones respecto de la media del ensamble.
    [xf_mean , xf_pert ] = mean_and_perts( xf )
    
    #Aplico la inflacion multiplicativa a la amplitud de las perturbaciones.
    xf_pert = xf_pert * Inflation
 
    #Calculo la matriz Pf
    Pf = np.cov( xf_pert )
   
    #Calculo la ganancia de Kalman. 
    #K = PH^t * inv( H P H^t + R )
    H = forward_operator_tl( xf )
    
    PHt=np.matmul( Pf  , H.transpose() )
    HPHt=np.matmul( H , PHt )
    HPHtinv=np.linalg.inv( HPHt + R )
   
    K = np.matmul( PHt , HPHtinv ) 
 

    xa = np.zeros( xf.shape )   
    #Calculo el update para cada miembro del ensamble
    for i in range( nens )  :
      #Calculo las observaciones perturbadas
      yo_pert = yo.transpose() + np.random.multivariate_normal( np.zeros( yo.shape ) , R )
      #Calculo el analisis para este miembro
      hxf = forward_operator( xf[:,i] )
      
      xa[:,i] = xf[:,i] + np.dot( K , ( yo_pert - hxf ) )
      
    
    [xa_mean , xa_pert ] = mean_and_perts( xa )
    Pa = np.cov( xa_pert ) 
   
    hxamean = forward_operator( xa_mean )
    hxfmean = forward_operator( xf_mean )
    OmB = yo - hxfmean
    OmA = yo - hxamean
   
    return xa , xa_mean , Pa , OmB , OmA
  
def analysis_update_EnSRF( yo , xf , forward_operator , R , Inflation )   :

   #---------------------------------------------#   
   #  Dada matrices R, un ensamble de campos preliminares y un conjunto de observaciones
   #  calcula el analisis, la matriz de covarianza de los errores del analisis
   #  y las perturbaciones para integrar el ensamble durante el tiempo siguiente.
   #
   #  Los metodos que usan esta funcion son el Ensemble Square Root Filter
   #  Whitaker and Hamill 2002
   #
   #
   #  input:
   #       yo      - observaciones
   #       xf(dim,nens) - campo preliminar (first gues)  ensemble
   #       forward_operator - operador de las observaciones (funcion)
   #       forward_operator_tl - tangente lineal del operador de las observaciones (funcion)
   #  output:
   #       xa(dim,nens) - analysis ensemble
   #
   #---------------------------------------------

   #En esta formulacion las perturbaciones del ensamble se actualizan de forma "deterministica"
   #esto es no hay necesidad de generar un conjunto de observaciones perturbadas estocasticamente.
   #esto remueve la componente estocastica y el filtro se vuelve totalmente determinista.
   #Por otra parte se hace una asimilacion secuencial de las observaciones. En lugar de asimilar
   #todas las observaciones al mismo timepo se asimila una observacion por vez.

    #Obtengo la cantidad de miembros en el ensamble
    [nvar , nens ]= xf.shape

    #Obtengo el numero de observaciones
    nobs = yo.shape[0]
      
    #Calculamos el ensamble en el espacio de las perturbaciones
    y=np.zeros((nobs,nens))
    
    #Defino la matriz que guardara las perturbaciones respecto de la media del ensamble.
    [xf_mean , xf_pert ] = mean_and_perts( xf )
    
    #Aplico la inflacion multiplicativa a la amplitud de las perturbaciones.
    xf_pert = xf_pert * Inflation
    
    for iens in range( nens ) :
        y[:,iens] = forward_operator( xf_mean + xf_pert[:,iens] )
    
    #Defino la media del ensamble y las perturbaciones en el espacio de las observaciones.
    [ymean , ypert ] = mean_and_perts( y )
    

    #Comienzo el loop sobre las observaciones
    
    xmean = np.copy( xf_mean )
    xpert = np.copy( xf_pert )
    
    dy = yo - forward_operator( xf_mean )  #Innovacion de la media del ensamble.
    
    #En el loop serial con cada observacion hacemos un update no solo del estado 
    #sino tambien del estado en el espacio de las observaciones
    #Esto evita que tengamos que aplicar nuevamente el operador de las observaciones
    #cada vez que asimilamos una nueva observacion. 
    #Por otra parte no necesitamos derivar explicitamente el tangente lineal del operador 
    #de las observaciones.
    for iobs in range( nobs )  :
        
        #HPHt + R
        HPHtR = np.dot( ypert[iobs,:].transpose() , ypert[iobs,:] ) * (1.0/(nens-1.0)) + R[iobs,iobs]
        PHt   = np.dot( xpert , ypert[iobs,:].transpose() ) * (1.0/(nens-1.0))
        K     = PHt * ( 1.0/ HPHtR )  #Si las observaciones se procesan serialmente entonces HPHtR es un escalar. 
        YYj   = np.dot( ypert , ypert[iobs,:].transpose() ) * (1.0/(nens-1.0))
        Ko    = YYj * ( 1.0 / HPHtR )  #Kalman gain para actualizar el estado en el espacio de las observaciones.
        #Calculamos alpha segun la ecuacion 13 de Whitaker y Hamill 
        alpha = 1.0 / ( 1.0 + np.sqrt(  R[iobs,iobs] / HPHtR ) )
  
        
        for iens in range( nens ) :        
           xpert[:,iens] = xpert[:,iens] - alpha * K * ypert[iobs,iens] 
           ypert[:,iens] = ypert[:,iens] - alpha * Ko * ypert[iobs,iens] 
        
        xmean = xmean +  K * dy[iobs] 
        dy    = dy    + alpha * Ko * dy[iobs] 
        
    xa = np.zeros( xf.shape )
    xa_mean = xmean
    xa_pert = xpert
    for iens in range( nens ):
        xa[:,iens] = xa_mean + xpert[:,iens]
        
    Pa = np.cov( xa_pert )
    
    #print(xf_mean,xa_mean)
   
    hxamean = forward_operator( xa_mean )
    hxfmean = forward_operator( xf_mean )
    OmB = yo - hxfmean
    OmA = yo - hxamean

    
    return xa , xa_mean , Pa , OmB , OmA   


def analysis_update_SIR( yo , xf , w_in , forward_operator , R , resampling = True )   :

   #---------------------------------------------#   
   #  Dada matrices R, un ensamble de campos preliminares y un conjunto de observaciones
   #  calcula  los pesos optimos y hace el resampling.
   #  Esta funcion se aplica al Importance Resampling Particle Filter
   #  SIR particle filter (Gordon et al 1993)
   #
   #
   #  input:
   #       yo      - observaciones
   #       xf(dim,nens) - campo preliminar (first gues)  ensemble
   #       w_in (nens)  - pesos de entrada
   #       forward_operator - operador de las observaciones (funcion)
   #  output:
   #       xa(dim,nens) - analysis ensemble
   #       w(dim,nens)  - weigths
   #
   #---------------------------------------------
    Effective_dimension = 1.0 
    
    #Obtengo la cantidad de miembros en el ensamble
    [nvar , nens ]= xf.shape
    
    Nthreshold = Effective_dimension * nens 
    #Calculo la inversa de la matriz de covarianza    
    Rinv = np.linalg.inv(R)
        
    #Calculamos los pesos en base al likelihood de las observaciones 
    #dada cada una de las particulas.
    w=np.zeros( w_in.shape )
    for iens in range( nens ) :
        yf = forward_operator( xf[:,iens] )
        w[iens] = w_in[iens] * np.exp( -0.5 * np.dot( (yo-yf).transpose() , np.dot( Rinv , yo - yf ) ) )

    #Normalizamos los pesos para que sumen 1.

    w = w / np.sum(w)
    
    xa=np.copy(xf)
    #Aplicamos el resampling siempre y cuando se verifique el criterio
    if resampling : 
    
       Neff=np.sum(w**2)**-1
       
       if Neff < Nthreshold: #threshold --> resampling
           indexres=resample(w)
           xa=xf[:,indexres]
           #Reseteamos el valor de los pesos.
           w=np.zeros(nens)+1.0/nens
    
    [xa_mean , xa_pert] = mean_and_perts( xa , weigthed=True , w=w )       
    Pa = np.cov( xa_pert ) 
   
    hxamean = forward_operator( xa_mean )
    hxfmean = forward_operator( np.mean(xf,1) )
    OmB = yo - hxfmean
    OmA = yo - hxamean

    return xa , xa_mean , Pa , OmB , OmA , w

def analysis_update_EMD( yo , xf , forward_operator , R , rtps_alpha , rejuv_param , multinf )   :

    #from emd import emd
    from scipy.spatial.distance import cdist
    import ot
   #---------------------------------------------#   
   #  Dada matrices R, un ensamble de campos preliminares y un conjunto de observaciones
   #  calcula  los pesos optimos y hace el resampling.
   #  Esta funcion se aplica al Earth Mover Distance Particle Filter
   #  Este filtro aplica una transformacion deterministica que es aquella que permite
   #  transformar el prior en el posterior con la minima redistribucion de masa (de probabilidad)
   #
   #  Esto es una transformacion que mueve las particulas una distancia minima tal que las particulas
   #  finales describan el posterior mientras que las particulas iniciales describen el prior.
   #
   #  Este filtro presenta una alternativa deterministica al resampling estocastico.
   #  Reich, S. (2013). A non-parametric ensemble transform method for bayesian inference.
   #  SIAM J Sci Comput, 35:A2013–A2014.
   #
   #  input:
   #       yo      - observaciones
   #       xf(dim,nens) - campo preliminar (first gues)  ensemble
   #       w_in (nens)  - pesos de entrada
   #       forward_operator - operador de las observaciones (funcion)
   #  output:
   #       xa(dim,nens)  - analysis ensemble
   #       w(nens)  - weigths
   #       S(nens,nens) - EM Flow matrix es la matriz que permite convertir el prior
   #       en el posterior.
   #
   #---------------------------------------------

    
    #Obtengo la cantidad de miembros en el ensamble
    [nvar , nens ]= xf.shape
    
    #Calculo la inversa de la matriz de covarianza    
    Rinv = np.linalg.inv(R)
    
    [xf_mean , xf_pert] = mean_and_perts( xf )
    
    xf_pert = multinf * xf_pert
    
    for i in range(nens)  :
        
        xf[:,i] = xf_mean + xf_pert[:,i]
        
    #Calculamos los pesos en base al likelihood de las observaciones 
    #dada cada una de las particulas.
    w=np.zeros( nens )
    for iens in range( nens ) :
        yf = forward_operator( xf[:,iens] )
        w[iens] = np.exp( -0.5 * np.dot( (yo-yf).transpose() , np.dot( Rinv , yo - yf ) ) )

    #Normalizamos los pesos para que sumen 1.

    w = w / np.sum(w)
    

    
    #La rutina que calcula la matriz de transformacion espera que cada fila sea un miembro del ensamble
    #y que cada columna sea una variable. En nuestro caso x viene al reves por eso tenemos que transponer
    #la matriz para realizar estos calculos.
    aux_xf = np.transpose(xf)
    aux_xf_pert = np.transpose(xf_pert)
    
    #Esta funcion de C resuelve el problema de la distancia minima obteniendo la matriz
    #de flujo S que permite convertir una muestra con distribucion igual al prior en otra muestra
    #con distribucion igual al posterior.
    #[distance , S ] = emd( aux_xf , np.copy(aux_xf)  , X_weights=np.ones(nens)/nens , Y_weights = w , return_flows= True) 
    D = cdist(aux_xf,aux_xf,'euclidean')
    S=ot.emd(np.ones(nens)/nens,w,D,numItermax=1.0e9,log=False)
    S = nens * S

    aux_xa=np.zeros(np.shape(aux_xf))
    
    tmp_rand = np.random.randn(nens,nens) / (nens-1)
    
    #Aplico en un solo paso la transformacion lineal y el rejuvenecimiento.
    
    aux_xa = np.dot( S , aux_xf ) + rejuv_param * np.dot( tmp_rand , aux_xf_pert  ) 
    
    xa=np.transpose(aux_xa)
    
    [xa_mean , xa_pert] = mean_and_perts( xa )
    
    #RTPS inflation.
    xa_std = np.sum( np.std(xa_pert,axis=1) )
    xf_std = np.sum( np.std(xf_pert,axis=1) )
    
    xa_pert = xa_pert * ( rtps_alpha * (xf_std - xa_std) / xa_std + 1)
    
    for i in range(0,nens) :
        xa[:,i] = xa_pert[:,i] + xa_mean
       
    Pa = np.cov( xa_pert ) 
    
    hxamean = forward_operator( xa_mean )
    hxfmean = forward_operator( np.mean(xf,1) )
    OmB = yo - hxfmean
    OmA = yo - hxamean

    return xa , xa_mean , Pa , OmB , OmA , w , S  



def analysis_update_ETKF( yo , xf , forward_operator , R , Inflation )   :

   #---------------------------------------------#   
   #  Dada matrices R, un ensamble de campos preliminares y un conjunto de observaciones
   #  calcula el analisis, la matriz de covarianza de los errores del analisis
   #  y las perturbaciones para integrar el ensamble durante el tiempo siguiente.
   #
   #  Los metodos que usan esta funcion son el Ensemble Square Root Filter
   #  Whitaker and Hamill 2002
   #
   #
   #  input:
   #       yo      - observaciones
   #       xf(dim,nens) - campo preliminar (first gues)  ensemble
   #       forward_operator - operador de las observaciones (funcion)
   #       forward_operator_tl - tangente lineal del operador de las observaciones (funcion)
   #  output:
   #       xa(dim,nens) - analysis ensemble
   #
   #---------------------------------------------

   #En esta formulacion se utiliza el Ensemble Transform Kalman Filter esto es
   #el update del analisis se calcula en el espacio del ensamble (esto es particularmente
   #ventajoso cuando la dimension del estado es mucho mayor que la cantidad de miembros en el ensamble)
   #Esta formulacion no requiere el calculo explicito del modelo tangente lineal
    #Obtengo la cantidad de miembros en el ensamble
    [nvar , nens ]= xf.shape

    #Obtengo el numero de observaciones
    nobs = yo.shape[0]
      
    #Calculamos el ensamble en el espacio de las perturbaciones
    y=np.zeros((nobs,nens))
    
    #Defino la matriz que guardara las perturbaciones respecto de la media del ensamble.
    [ xf_mean , xf_pert ] = mean_and_perts( xf )
    
    #Aplico la inflacion multiplicativa a la amplitud de las perturbaciones.
    xf_pert = xf_pert * Inflation
    
    for iens in range( nens ) :
        y[:,iens] = forward_operator( xf_mean + xf_pert[:,iens] )
    
    #Defino la media del ensamble y las perturbaciones en el espacio de las observaciones.
    [ymean , ypert ] = mean_and_perts( y )

    dy = yo - forward_operator( xf_mean )  #Innovacion de la media del ensamble.
    
    Rinv= np.linalg.inv(R) 

    # analysis error covariance in ensemble space

    Pahat=np.linalg.inv( np.dot( ypert.transpose() , np.dot( Rinv , ypert ) ) + (nens-1.0)*np.identity(nens) )

    # weight to update ensemble mean
    wabar = np.dot( Pahat , np.dot( ypert.transpose() , np.dot( Rinv , dy ) ) )

    # weight to update ensemble perturbations
    Wa = linalg.sqrtm( (nens-1)*Pahat )
   
    xa_mean = xf_mean + np.dot( xf_pert , wabar )
        
    xa_pert = np.dot( xf_pert , Wa )
    
    xa=np.zeros( xf.shape )
        
    for iens in range( nens ) :

       xa[:,iens]=xa_mean+xa_pert[:,iens]

    Pa = np.cov( xa_pert ) 
   
    hxamean = forward_operator( xa_mean )
    hxfmean = forward_operator( xf_mean )
    OmB = yo - hxfmean
    OmA = yo - hxamean

    return xa , xa_mean , Pa , OmB , OmA   


def analysis_update_3DVAR( yo , xf , P , forward_operator , forward_operator_tl , R )   :

   #---------------------------------------------
   #  Dada matrices R, P, un campo preliminar y un conjunto de observaciones
   #  calcula el analisis mediante una minimizacion de la funcion de costo. 
   #
   #  Los metodos que usan esta funcion son el 3DVAR
   #
   #  input:
   #       yo      - observaciones
   #       xf(dim) - campo preliminar (first gues)
   #       Pf      - matriz de covarianza de los errores del campo preliminar
   #       R       - matriz de covarianza de los errores de las observaciones
   #       forward_operator - funcion que pasa del espacio del estado al espacio de las observaciones
   #       forward_operator_tl - tangente lineal del forward opeator
   #  output:
   #       xa(dim) - analysis
   #
   #---------------------------------------------
   
   # Implementacion del gradiente descendente para 3DVAR.
   #Adaptado de https://stackoverflow.com/questions/17784587/gradient-descent-using-python-and-numpy
   max_iter = 1000           #Maxima cantidad de iteraciones
   alfa= 0.015               #Tasa de aprendizaje
   tolerancia=1.e-6          #Criterio de corte de la minimizacion.
   
   #La funcion de costo requiere calcular la inversa de P y R (en este caso
   #como el sistema tiene solo 3 variables podemos obtener esas inversas
   #explicitamente)
   invP = np.linalg.inv( P ) 
   invR = np.linalg.inv( R )

   #A continuacion viene la iteracion para obtener el x que minimiza la funcion de costo
   #Por conveniencia el calculo de la funcion de costo y su gradiente se definieron en funciones aparte.
   J=0

   #Inicializo x, la innovacion y el update (xb-xa)
   x=np.copy(xf) 
   innovation= yo - forward_operator(x)
   update = x - xf
   
   print('')
       
   for i in range( max_iter )  :

       H= forward_operator_tl( x )
       
       if i == 0 :
           [Jold , Jbold , Joold]= J3DVAR( update , innovation , invP , invR )
           print('J inicial: J ',Jold,' Jb ',Jbold,' Jo ',Joold)
       else      :
           Jold = J
           
       #Calculo el gradiente de la funcion de costo
       Jgrad = JGrad3DVAR( update , innovation , H , invP , invR )
       #print(Jgrad)

       #Actualizo x, la innovacion y el update
       x = x - alfa * Jgrad        
       innovation= yo - forward_operator(x)
       update = x - xf
       
       #Recalculo la funcion de costo
       [J , Jb , Jo ] = J3DVAR( update , innovation , invP , invR )
       
       #Si el cambio en la funcion de costo es muy pequeño no seguimos iterando
       if ( np.abs( J - Jold ) < tolerancia ) :
           break
       
   xa = x  #xa se obtiene como el x que minimiza la funcion de costo
   print('J final: J ',J,' Jb ',Jb,' Jo ',Jo)

   print('N iteraciones: ',i+1)
   
   print('')
   
   OmB=yo-  forward_operator( xf )    #To compute O-B statistics
   OmA=yo - forward_operator( xa )    #To compute A-B statistics
   
   return xa , OmB , OmA 

def analysis_update_4DVAR( yo , xf , P , parameters , dt , bst , forward_model, tl_model , forward_operator , forward_operator_tl , R )   :

   #---------------------------------------------
   #  Dada matrices R, P, un campo preliminar y un conjunto de observaciones
   #  calcula el analisis mediante una minimizacion de la funcion de costo. 
   #
   #  Los metodos que usan esta funcion son el 4DVAR
   #
   #  input:
   #       yo      - observaciones
   #       xf(dim) - campo preliminar (first gues)
   #       Pf      - matriz de covarianza de los errores del campo preliminar
   #       R       - matriz de covarianza de los errores de las observaciones
   #       forward_model    - funcion que integra el modelo no lineal
   #       tl_model         - funcion que permite obtener el modelo tangente lineal 
   #       a partir del estado del sistema. 
   #       forward_operator - funcion que pasa del espacio del estado al espacio de las observaciones
   #       forward_operator_tl - tangente lineal del forward opeator
   #  output:
   #       x_best(dim) - es la trayectoria no lineal (consistente con las ecuaciones del modelo)
   #       que mejor aproxima a las observaciones dentro de la ventana de asimilacion.
   #
   #       Notar que en este caso la rutina no solo calcula el analisis sino que tambien calcula
   #       la trayectoria no lineal que mejor aproxima a las observaciones en toda la ventana
   #       de asimilacion.
   #       Esta funcion esta preparada para que el array que contiene a las observaciones
   #       contenga varios tiempos. 
   #---------------------------------------------
   
   # Implementacion del gradiente descendente para 3DVAR.
   #Adaptado de https://stackoverflow.com/questions/17784587/gradient-descent-using-python-and-numpy
   max_iter = 1000           #Maxima cantidad de iteraciones
   alfa= 0.015               #Tasa de aprendizaje
   tolerancia=1.e-6          #Criterio de corte de la minimizacion.
   
   [ntimes , nobs ] = np.shape(yo)

      
   #La funcion de costo requiere calcular la inversa de P y R (en este caso
   #como el sistema tiene solo 3 variables podemos obtener esas inversas
   #explicitamente)
   invP = np.linalg.inv( P ) 
   invR = np.linalg.inv( R )

   #A continuacion viene la iteracion para obtener el x que minimiza la funcion de costo
   #Por conveniencia el calculo de la funcion de costo y su gradiente se definieron en funciones aparte.

   #Inicializo x, la innovacion y el update (xb-xa)
   x=np.copy(xf) 

   
   #Recordar que en 4DVAR se optimiza el estado al inicio de la ventana de 
   #asimilacion. 
       
   for i in range( max_iter )  :
       
       if i == 0 :
           Jold = 1.0e99
       else:
           Jold = J    
       
       #Calculo la funcion de costo y su gradiente
       [J , Jb , Jo , Jgrad]= JGrad4DVAR( xf , x , parameters , dt , bst , yo , invP , invR , forward_model , tl_model , forward_operator , forward_operator_tl )

       #print('J inicial: J ',J,' Jb ',Jb,' Jo ',Jo)
       #print('JGrad inicial',Jgrad)

       
       #Actualizo x, la innovacion y el update
       x = x - alfa * Jgrad  

       #Si el cambio en la funcion de costo es muy pequeño no seguimos iterando
       if ( np.abs( J - Jold ) < tolerancia ) :
           break
   
   #print('J final: J ',J,' Jb ',Jb,' Jo ',Jo)
   #print('N iteraciones: ',i+1)
   #print('')
   
   #Integramos el modelo no-lineal para obtener la trayectoria que mejor aproxima
   #a las observaciones a lo largo de la ventana de asimilacion. Esta trayectoria
   #se obtiene tomando la mejor estimacion del estado al inicio de la ventana (xa)
   #e integrando el modelo no lineal a lo largo de toda la ventana.
   
   x_best = np.zeros((ntimes+1,np.size(x)))
   x_best[0,:] = x[:]
   
   for it in range( ntimes ) :
      for i in range( int(bst) ):
         #Calculamos el propagante del modelo tangente lineal en toda la ventana.
         #Calculamos la trayectoria hacia adelante usando el modelo no lineal
         x = forward_model( x , parameters , dt )
      
      x_best[it+1,:] = x 
   
   return x_best  

   
def mean_and_perts( xens , weigthed=False , w=None ) :
    
    [nvar,nens]=np.shape(xens)
    
    if weigthed  :
       mean=np.average(xens,axis=1,weights=w)
    else  :
       mean=np.mean(xens,1)
    
    perts=np.zeros( (nvar,nens) )
    for iens in range( nens ) :
        perts[:,iens] = xens[:,iens] - mean 
        
    return mean , perts

def JGrad3DVAR( update , innovation , H , invP , invR ) :

    # nablaJ = 2 inv(B) ( x-xb) - 2H' inv(R) ( y-h(x)) 
    nablaJ = 2 * np.dot( invP , update ) - 2* np.dot( H.transpose() , np.dot( invR , innovation ) )

    return nablaJ

def J3DVAR( update , innovation , invP , invR ) :
#    # J = (x - xb)' inv(P) (x - xb) + (y-h(x))'inv(R)(y-h(x))
     #                Jb                          Jo
      
    Jo= np.dot( innovation.transpose() , np.dot( invR , innovation ) )
    Jb= np.dot( update.transpose() , np.dot( invP , update ) )
    J = Jb + Jo  
    
    return J , Jb , Jo


def JGrad4DVAR( xb , x , parameters , dt , bst , yo , invP , invR , forward_model , tl_model , forward_operator , forward_operator_tl ) :
    
   #Calculamos la funcion de costo del 4DVAR y su gradiente
   #Utilizamos siempre una integracion del modelo no lineal (lo cual es costoso)
   #Asumimos que el operador de las observaciones no se modifica de un tiempo al tiempo siguiente.
   
   #En la practica no siempre se actualiza la trayectoria no lineal integrando el modelo.
   #es posible usar el modelo tangente lineal (mas barato computacionalmente) para calcular
   #la evolucion de la perturbacion dentro de la ventana de asimilacion. 
   
   update = x - xb
      
   Jb = np.dot( update.transpose() , np.dot( invP , update ) )
   
   nablaJ = 2.0 * np.dot( invP , update )
   
   Jo=0.0
   
   [ntimes , nobs ] = np.shape(yo)
   
   xf = np.copy( x )
   L_total=np.identity(3)  #Inicializo el propagante del modelo tangente lineal.
   for it in range( ntimes ) :
       
       for i in range( int(bst) ):
           #Calculamos el propagante del modelo tangente lineal en toda la ventana.
           #Calculamos el propagante del modelo tangente lineal en toda la ventana.
           L = tl_model( xf , parameters , dt )
           L_total = np.dot( L , L_total )

           #Calculamos la trayectoria hacia adelante usando el modelo no lineal
           xf = forward_model( xf , parameters , dt )

          
       innovation = yo[it,:] - forward_operator( xf )

       Jo = Jo + np.dot( innovation.transpose() , np.dot( invR , innovation ) )
 
       H = forward_operator_tl( xf )
       
       nablaJ = nablaJ - 2.0 * np.dot( L_total.transpose() , np.dot( H.transpose() , np.dot( invR , innovation ) ) ) 
           
   J = Jb + Jo

   return J , Jb , Jo , nablaJ
    


def resample(weights):
    """ Performs the residual resampling algorithm used by particle filters.
    Taken from pyfilt 0.1?
    Parameters:
    ----------
    weights  
    Returns:
    -------
    indexes : ndarray of ints
    array of indexes into the weights defining the resample.
    """

    N = len(weights)
    indexes = np.zeros(N, 'i')
        
    # take int(N*w) copies of each weight, which ensures particles with the
    # same weight are drawn uniformly
    num_copies = (np.floor(N*np.asarray(weights))).astype(int)
    k = 0
    for i in range(N):
        for _ in range(num_copies[i]): # make n copies
            indexes[k] = i
            k += 1
                
    # use multinormal resample on the residual to fill up the rest. 
    residual = weights - num_copies     # get fractional part
    residual /= sum(residual)           # normalize
    cumulative_sum = np.cumsum(residual)
    cumulative_sum[-1] = 1. # avoid round-off errors: ensures sum is exactly one
    ran1=np.random.uniform(size=N-k)
    indexes[k:N] = np.searchsorted(cumulative_sum, ran1)

    return indexes
   

def analysis_verification( forward_operator , da_exp ) :
    
   #---------------------------------------------
   #  Esta funcion calcula el RMSE y BIAS del analisis, del first guess y de 
   #  las observaciones.
   #---------------------------------------------
   spinup=200
   statea=da_exp['statea'][spinup:,:]
   statef=da_exp['statef'][spinup:,:,:]
   state =da_exp['state'][spinup:,:]
   yobs  =da_exp['yobs'][spinup:,:]

   da_exp['rmse_a']= np.sqrt( np.nanmean( np.power( statea - state , 2 ) , 0 ) )
 

   da_exp['bias_a']= np.nanmean( statea - state , 0 ) 
   
   da_exp['rmse_f']=np.zeros( (np.size(da_exp['rmse_a']),da_exp['forecast_length']) )
   da_exp['bias_f']=np.zeros( (np.size(da_exp['rmse_a']),da_exp['forecast_length']) )

   #Calculamos el error en el pronostico (incluyendo el first guess)
   for j in range( da_exp['forecast_length'] ) :
      da_exp['rmse_f'][:,j]= np.sqrt( np.nanmean( np.power( statef[:,:,j] - state , 2 ) , 0 ) )
      da_exp['bias_f'][:,j]= np.nanmean( statef[:,:,j] - state , 0 ) 

   
   tmpobs = np.zeros( np.shape( yobs ) )
   for it in range( np.shape( state )[0] ) :
      tmpobs[it,:] = forward_operator( state[it,:] )    
       
   da_exp['rmse_o'] = np.sqrt( np.nanmean( np.power( yobs - tmpobs , 2 ) , 0 ) )
   da_exp['bias_o'] = np.nanmean( yobs - tmpobs , 0 )
   

   #Para calcular el error de las obs genero un nuevo conjunto de observaciones pero sin error.
   tmpobs = np.zeros( np.shape( yobs ) )
   nobs=da_exp['nobs']
   da_exp['rmse_fo'] = np.zeros( (nobs , da_exp['forecast_length'] ) )
   da_exp['bias_fo'] = np.zeros( (nobs , da_exp['forecast_length'] ) )
   for j in range( da_exp['forecast_length'] )  :
      for it in range( np.shape( state )[0] ) :
         tmpobs[it,:] = forward_operator( statef[it,:,j] )    
      da_exp['rmse_fo'][:,j] = np.sqrt( np.nanmean( np.power( yobs - tmpobs , 2 ) , 0 ) )
      da_exp['bias_fo'][:,j] = np.nanmean( yobs - tmpobs , 0 )
   tmpobs = np.zeros( np.shape( yobs ) )
   for it in range( np.shape( state )[0] ) :
      tmpobs[it,:] = forward_operator( statea[it,:] )    
   da_exp['rmse_ao'] = np.sqrt( np.nanmean( np.power( yobs - tmpobs , 2 ) , 0 ) )
   da_exp['bias_ao'] = np.nanmean( yobs - tmpobs , 0 )

   print()
   print()

   print('Analysis RMSE: ',da_exp['rmse_a'])
   print('First guess RMSE: ',da_exp['rmse_f'][:,0])
   print('Obaservations RMSE: ',da_exp['rmse_o'])

   print()
   print()

   print('Analysis BIAS: ',da_exp['bias_a'])
   print('First guess BIAS: ',da_exp['bias_f'][:,0])
   print('Observations BIAS: ',da_exp['bias_o'])

   print()
   print()

   return da_exp

#------------------------------------------------------------------------------
# FIGURAS
#------------------------------------------------------------------------------
   
def state_plot( da_exp ) :
   import matplotlib.pyplot as plt
#   from mpl_toolkits.mplot3d import Axes3D
   
#   print('Evolucion del estado verdadero en 3D')
   
#   #Graficamos la evolucion del modelo en 3D.
#   state=da_exp['state']
#   plt.figure()
#   ax  = plt.axes(projection="3d")
#   ax.plot3D(state[:,0],state[:,1],state[:,2],'blue')
   
#   plt.savefig( da_exp['main_path'] + '/figs/' + da_exp['exp_id'] + '_3DTruePlot.png')
   
   
def forecast_error_plot( da_exp ) :
   import matplotlib.pyplot as plt
   
   spinup=200
   
   print('Evolucion del RMSE y bias del pronostico con el plazo de pronostico')
   plt.figure()
   plt.plot(np.sum(da_exp['rmse_f'],0),'b',label='RMSE')
   plt.plot(np.sum(da_exp['bias_f'],0),'b--',label='Bias')
   
   if np.size( np.shape( da_exp['P'] ) ) == 4  :  #Veo si tengo una estimacion del error del pronostico.
       Ptracesq = np.nanmean( np.sqrt(da_exp['P'][spinup:,0,0,:]) + np.sqrt(da_exp['P'][spinup:,1,1,:]) + np.sqrt(da_exp['P'][spinup:,2,2,:]) , 0 )
       plt.plot( Ptracesq ,'r',label='Ptrace_sq')
       
   
   plt.grid()
   plt.legend()
   
   
   
   plt.savefig( da_exp['main_path'] + '/figs/' + da_exp['exp_id'] + '_ForecastRMSEBias.png')
      

def state_evolution( da_exp , ini_time , end_time )  :
    import matplotlib.pyplot as plt
    #Grafica las el analisis, el first guess y las observaciones para un periodo
    #de tiempo comprendido entre ini_time y end_time 
    print('Evolucion de la variable del analisis, el first guess, el nature run y las observaciones')    
    
    for ivar in range( da_exp['nvars'] )  :
       
       plt.figure()
       plt.title('Variable ' + str(ivar) )
       
       
       plt.plot(da_exp['state'][ini_time:end_time,ivar],'g',label='True')
       plt.plot(da_exp['statea'][ini_time:end_time,ivar],'r',label='Analysis')
       plt.plot(da_exp['statef'][ini_time:end_time,ivar,0],'b',label='First guess')
       
       plt.legend()
       plt.xlabel('Ciclos de asimilacion')
       plt.ylabel('Variable')
       plt.grid()

       plt.savefig( da_exp['main_path'] + '/figs/' + da_exp['exp_id'] + '_StateTimeSeries_Var'+ str(ivar) + '.png')
  
def obs_evolution( da_exp , ini_time , end_time , forward_operator )  :
    
    import matplotlib.pyplot as plt
    #Grafica las el analisis, el first guess y las observaciones para un periodo
    #de tiempo comprendido entre ini_time y end_time 
    print('Evolucion de la variable del analisis, el first guess, el nature run y las observaciones')    
    
    tmp_a=da_exp['statea'][ini_time:end_time,:]
    tmp_f=da_exp['statef'][ini_time:end_time,:,0]
    tmp_t=da_exp['state'][ini_time:end_time,:]

    ntimes=tmp_a.shape[0]
    
    tmp_ao=np.zeros((ntimes,da_exp['nobs']))
    tmp_fo=np.zeros((ntimes,da_exp['nobs']))
    tmp_to=np.zeros((ntimes,da_exp['nobs']))
    tmp_o=da_exp['yobs'][ini_time:end_time,:]
    
    for it in range( ntimes ) :
       tmp_ao[it,:]=forward_operator( tmp_a[it,:] )
       tmp_fo[it,:]=forward_operator( tmp_f[it,:] )
       tmp_to[it,:]=forward_operator( tmp_t[it,:] )
    
    for iobs in range( da_exp['nobs'] )  :
       
       plt.figure()
       plt.title('Obs ' + str(iobs) )
       
       plt.plot(tmp_o[:,iobs],'ko',label='Obs')
       plt.plot(tmp_to[:,iobs],'g',label='True')
       plt.plot(tmp_ao[:,iobs],'r',label='Analysis')
       plt.plot(tmp_fo[:,iobs],'b',label='First guess')
       
       plt.legend()
       plt.xlabel('Ciclos de asimilacion')
       plt.ylabel('Variable')
       plt.grid()

       plt.savefig( da_exp['main_path'] + '/figs/' + da_exp['exp_id'] + '_ObsTimeSeries_Obs'+ str(iobs) + '.png')
    
    
def error_evolution( da_exp , ini_time , end_time )  :
    
    import matplotlib.pyplot as plt
    #Grafica las el error del analisis, del first guess y de las observaciones para un periodo
    #de tiempo comprendido entre ini_time y end_time 
    print('Evolucion del error del analisis, el first guess, el nature run y las observaciones')    
    
    for ivar in range(da_exp['nvars'])  :
       plt.figure()
       plt.title('Variable ' + str(ivar) )
       
       
       plt.plot(da_exp['statea'][ini_time:end_time,ivar]-da_exp['state'][ini_time:end_time,ivar],'r',label='Ana. Error')
       plt.plot(da_exp['statef'][ini_time:end_time,ivar,0]-da_exp['state'][ini_time:end_time,ivar],'b',label='F.G. Error')
       plt.plot(da_exp['statef'][ini_time:end_time,ivar,0]-da_exp['statea'][ini_time:end_time,ivar],'g',label='Update')
       
       plt.legend()
       plt.grid()
       plt.xlabel('Ciclos de asimilacion')
       plt.ylabel('Error')

       plt.savefig( da_exp['main_path'] + '/figs/' + da_exp['exp_id'] + '_ErrorTimeSeries_Var'+ str(ivar) + '.png')
       
       
       
       
def rmse_evolution( da_exp , ini_time , end_time )  :
    
    import matplotlib.pyplot as plt
    #Grafica las el error del analisis, del first guess y de las observaciones para un periodo
    #de tiempo comprendido entre ini_time y end_time 
    print('Evolucion temporal del RMSE')    
    
    plt.figure()
    
       
    rmse_t_f = np.sqrt(np.mean( np.power( da_exp['statef'][ini_time:end_time,:,0]-da_exp['state'][ini_time:end_time,:] , 2) , 1))
    rmse_t_a = np.sqrt(np.mean( np.power( da_exp['statea'][ini_time:end_time,:]-da_exp['state'][ini_time:end_time,:] , 2) , 1))
    if np.size( np.shape( da_exp['P'] ) ) == 4 :
       p_sqtrace = np.sqrt( (da_exp['P'][ini_time:end_time,0,0,0] + da_exp['P'][ini_time:end_time,1,1,0] + da_exp['P'][ini_time:end_time,2,2,0] )/3 )   
       plt.plot(p_sqtrace,'k',label='P sqrt trace')
    if np.size( np.shape( da_exp['P'] ) ) == 3 :
       p_sqtrace = np.sqrt( (da_exp['P'][ini_time:end_time,0,0] + da_exp['P'][ini_time:end_time,1,1] + da_exp['P'][ini_time:end_time,2,2] )/3 )   
       plt.plot(p_sqtrace,'k',label='P sqrt trace')
    
    plt.plot(rmse_t_a,'r',label='Ana. RMSE')
    plt.plot(rmse_t_f,'b',label='F.G. RMSE')
    
    plt.legend()
    plt.xlabel('Ciclos de asimilacion')
    plt.ylabel('RMSE')
    plt.grid()
       
    plt.savefig( da_exp['main_path'] + '/figs/' + da_exp['exp_id'] + '_TotalRMSETimeSeries.png')  
    
    
def analysis_rmse_3d( da_exp ) :
    
    import plotly.graph_objects as go
    from plotly.offline import plot
    
    spinup=200

    
    fig = go.Figure(data=[go.Scatter3d(
            x=da_exp['state'][spinup:,0] ,
            y=da_exp['state'][spinup:,1] ,            
            z=da_exp['state'][spinup:,2] ,            
            mode='markers' ,
            marker=dict(
                    size=5,
                    color=np.sqrt( np.sum( np.power( da_exp['state'][spinup:,:] - da_exp['statef'][spinup:,:,0] ,2 ) , 1 ) ) , 
                    colorscale='Viridis',
                    opacity=0.8
                        )
            )])
       
    plot(fig)  
    
def Ptrace_3d( da_exp ) :
    
    import plotly.graph_objects as go
    from plotly.offline import plot
    
    spinup=200
    
    if np.size( np.shape( da_exp['P'] ) ) == 4 :
       p_sqtrace = np.sqrt( (da_exp['P'][spinup:,0,0,0] + da_exp['P'][spinup:,1,1,0] + da_exp['P'][spinup:,2,2,0] )/3 )   
    if np.size( np.shape( da_exp['P'] ) ) == 3 :
       p_sqtrace = np.sqrt( (da_exp['P'][spinup:,0,0] + da_exp['P'][spinup:,1,1] + da_exp['P'][spinup:,2,2] )/3 )   

    
    fig = go.Figure(data=[go.Scatter3d(
            x=da_exp['state'][spinup:,0] ,
            y=da_exp['state'][spinup:,1] ,            
            z=da_exp['state'][spinup:,2] ,            
            mode='markers' ,
            marker=dict(
                    size=5,
                    color= p_sqtrace , 
                    colorscale='Viridis',
                    opacity=0.8
                        )
            )])
       
    plot(fig)   
    
       
def save_exp( da_exp )   :
    import pickle
    
    filename = da_exp['main_path'] + '/data/' + da_exp['exp_id'] + '_ANALYSIS_EXP_DATA.pkl'
    outfile = open(filename,'wb')

    pickle.dump( da_exp , outfile )
    outfile.close()


def directory_init( da_exp ) :
    
    import os
    
    try : 
       os.mkdir( da_exp['main_path'] )
       os.mkdir( da_exp['main_path'] + '/figs/' )
       os.mkdir( da_exp['main_path'] + '/data/' )
    except :
       print('WARNING: el directorio ya existe')






