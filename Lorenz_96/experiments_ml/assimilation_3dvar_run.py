# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:36:06 2017

@author: RISDA 2018
"""

#Run a LETKF experiment using the observations created by the script run_nature.py

import sys
sys.path.append('../model/')
sys.path.append('../data_assimilation/')

from model  import lorenzn          as model          #Import the model (fortran routines)
from obsope import common_obs       as hoperator      #Import the observation operator (fortran routines)
from da     import common_da_tools  as das            #Import the data assimilation routines (fortran routines)

import matplotlib.pyplot as plt
import numpy as np
import time
import assimilation_conf_1scale3DVAR as conf         #Load the experiment configuration
#import assimilation_conf_2scale_F16 as conf         #Load the experiment configuration
#import assimilation_conf_2scale_F20 as conf         #Load the experiment configuration
from scipy import stats
import os

#=================================================================
# LOAD CONFIGURATION : 
#=================================================================

GeneralConf=conf.GeneralConf
DAConf     =conf.DAConf
ModelConf  =conf.ModelConf

#=================================================================
#  LOAD OBSERVATIONS AND NATURE RUN CONFIGURATION
#=================================================================

print('Reading observations from file ',GeneralConf['ObsFile'])

InputData=np.load(GeneralConf['ObsFile'],allow_pickle=True)

ObsConf=InputData['ObsConf'][()]

YObs    =  InputData['YObs']         #Obs value
ObsLoc  =  InputData['ObsLoc']       #Obs location (space , time)
ObsType =  InputData['ObsType']      #Obs type ( x or x^2)
ObsError=  InputData['ObsError']     #Obs error 

#If this is a twin experiment copy the model configuration from the
#nature run configuration.
if DAConf['Twin']   :
  print('')
  print('This is a TWIN experiment')
  print('')
  ModelConf=InputData['ModelConf'][()]
  
#Times are measured in number of time steps. It is important to keep
#consistency between dt in the nature run and inthe assimilation experiments.
ModelConf['dt'] = InputData['ModelConf'][()]['dt']

#Store the true state evolution for verfication 
XNature = InputData['XNature']   #State variables
CNature = InputData['CNature']   #Parameters
FNature = InputData['FNature']   #Large scale forcing.

#=================================================================
# INITIALIZATION : 
#=================================================================

#We set the length of the experiment according to the length of the 
#observation array.
DALength = int( max( ObsLoc[:,1] ) / DAConf['Freq'] ) 

#DALength = 3

#Get the number of parameters
NCoef=ModelConf['NCoef']
#Get the size of the state vector
Nx=ModelConf['nx']
#Get the size of the small-scale state
NxSS=ModelConf['nxss']
#Get the number of ensembles
NEns=1   #En 3DVAR el pronostico es deterministico (no hay ensambles involucrados).

#Memory allocation and variable definition.

XA=np.zeros([Nx,NEns,DALength])                         #Analisis ensemble
XF=np.zeros([Nx,NEns,DALength])                         #Forecast ensemble
XFNN=np.zeros([Nx,NEns,DALength])                        #El forecast corregido por la red neuronal. 
PA=np.zeros([Nx,NEns,NCoef,DALength])                   #Analized parameters
PF=np.zeros([Nx,NEns,NCoef,DALength])                   #Forecasted parameters

F=np.zeros([Nx,NEns,DALength])                          #Total forcing on large scale variables.

#Initialize model configuration, parameters and state variables.
if not ModelConf['EnableSRF']    :
  XSigma=0.0
  XPhi=1.0
else                             :
  XSigma=ModelConf['XSigma']
  XPhi  =ModelConf['XPhi']

if not ModelConf['EnablePRF']    :
  CSigma=np.zeros(NCoef)
  CPhi=1.0
else                     :
  CSigma=ModelConf['CSigma']
  CPhi  =ModelConf['CPhi']


if not ModelConf['FSpaceDependent'] :
  FSpaceAmplitude=np.zeros(NCoef)
else                   :
  FSpaceAmplitude=ModelConf['FSpaceAmplitude']

FSpaceFreq=ModelConf['FSpaceFreq']

#Initialize random forcings
CRF=np.zeros([NEns,NCoef])
RF =np.zeros([Nx,NEns])

#Initialize small scale variables and forcing
XSS=np.zeros((NxSS,NEns))
SFF=np.zeros((Nx,NEns))

C0=np.zeros((NCoef,Nx,NEns))

#Generate a random initial conditions and initialize deterministic parameters
for ie in range(0,NEns)  :

   XA[:,ie,0]=ModelConf['Coef'][0]/2 + DAConf['InitialXSigma'] * np.random.normal( size=Nx )

   for ic in range(0,NCoef) : 
#       if DAConf['ParameterLocalizationType']==3 :
#           PA[:,ie,ic,0]=ModelConf['Coef'][ic] + DAConf['InitialPSigma'][ic] * np.random.normal( size=Nx )
#       else                                      :
           PA[:,ie,ic,0]=ModelConf['Coef'][ic] + DAConf['InitialPSigma'][ic] * np.random.normal( size=1 )
           
#=================================================================
#  MAIN DATA ASSIMILATION LOOP : 
#=================================================================
start_cycle = time.time()

for it in range( 1 , DALength  )         :

  
   print('Data assimilation cycle # ',str(it) )

   #=================================================================
   #  ENSEMBLE FORECAST  : 
   #=================================================================   

   #Run the ensemble forecast
   #print('Runing the ensemble')

   ntout=int( 2*DAConf['Freq'] / DAConf['TSFreq'] ) + 1  #Output the state every ObsFreq time steps.
   
   #Esta llamada a la funcion hay que modificarla para que integre los primeros 3 tiempos.
   [ XFtmp , XSStmp , DFtmp , RFtmp , SSFtmp , CRFtmp, CFtmp ]=model.tinteg_rk4( nens=NEns  , nt=DAConf['Freq']*2 ,  ntout=ntout ,
                                           x0=XA[:,:,it-1]     , xss0=XSS , rf0=RF    , phi=XPhi     , sigma=XSigma,
                                           c0=PA[:,:,:,it-1]   , crf0=CRF             , cphi=CPhi    , csigma=CSigma, param=ModelConf['TwoScaleParameters'] , 
                                           nx=Nx,  nxss=NxSS   , ncoef=NCoef  , dt=ModelConf['dt']   , dtss=ModelConf['dtss'])

   PF[:,:,:,it] = CFtmp[:,:,:,-2]       #Store the parameter at the end of the window. 
   XF[:,:,it]=XFtmp[:,:,-2]             #Store the state variables ensemble at the end of the window.

   F[:,:,it] =DFtmp[:,:,-2]+RFtmp[:,:,-2]+SSFtmp[:,:,-2]  #Store the total forcing 
   
   XSS=XSStmp[:,:,-2]
   CRF=CRFtmp[:,:,-2]
   RF=RFtmp[:,:,-2]
   
   #print('Ensemble forecast took ', time.time()-start, 'seconds.')
   
   #Definimos el pronostico corregido por la red (o sin corregir segun corresponda).
   XFNN[:,:,it]= np.copy( XF[:,:,it] ) #En el 3DVAR clasico, no hay red que corrija el error sistematico.
   
   ##Ojo! solo podemos comparar bien con las observaciones al final de la ventana. 
   #XFtmp = np.repeat( XFNN[:,:,it][:,:,np.newaxis] , ntout , 2 )

   #=================================================================
   #  GET THE OBSERVATIONS WITHIN THE TIME WINDOW  : 
   #=================================================================

   #Screen the observations and get only the onew within the da window
   window_mask= ObsLoc[:,1] == it * DAConf['Freq'] 
 
   ObsLocW=ObsLoc[window_mask,:]                                     #Observation location within the DA window.
   ObsTypeW=ObsType[window_mask]                                     #Observation type within the DA window
   YObsW=YObs[window_mask]                                           #Observations within the DA window
   NObsW=YObsW.size                                                  #Number of observations within the DA window
   ObsErrorW=ObsError[window_mask]                                   #Observation error within the DA window  


   #=================================================================
   #  OBSERVATION OPERATOR  : 
   #================================================================= 
   #Obtengo la matriz H (asumo que el operador H es lineal)
   H = np.zeros( ( NObsW , Nx ) )
   for io in range( NObsW ) :
       H[io,int(ObsLocW[io,0]-1.0)] = 1.0
  
   #=================================================================
   #  LETKF DA  : 
   #================================================================= 

   #print('Data assimilation')

   #STATE VARIABLES ESTIMATION:
  
   #start = time.time()
   
   #Definir como se construye la matriz de covarianza de los errores del forecast (P).
   #Matriz constante para 3DVAR
   #Guardamos las matrices de covarianza estimadas a partir de los pronosticos en un archivo y 
   #P = np.load ( .... )  #podemos guardar la matriz de covarianza y la de correlacion.
   P = 0.5*np.eye( Nx ) 
   #Matriz de correlacion constante pero con varianza estimada por la red.
   # diagP = viene de la Red 
   # corr_mat = np.load( ... ) construimos la matriz de covarianza a partir de diagP y de corr_mat. 
   #  P_i_j = np.sqrt( diagP )_i * np.sqrt( diagP )_j * corr_mat_i_j  (esto es el producto externo de diagP con si mismo por corr_mat)
   #Matriz de covarianza con estructura variable usando proyeccion en el espacio de PCA de los errores (Pierre). 
       
   #Asumo que a partir de aca vamos a tener definido P.
       
    
   #Implementar el calculo del analisis en este caso. 
   # x_a = x_f + P H^t * ( H P H^t + R )^-1 ( y_o - H( x_f ) )

   # R siempre lo asumimos diagonal. 
   R=np.diag( np.squeeze(ObsErrorW) )
    
   HPHR = np.matmul( H , np.matmul( P , H.T ) ) + R 
   Xf = np.repeat( XF[:,0,it][:,np.newaxis] , 1 , 1)  #Genero un vector columna.
   Innov = YObsW - np.dot( H , Xf )
   XA[:,0,it] = np.squeeze( Xf + np.matmul( np.matmul( np.matmul( P , H.T ) , np.linalg.inv( HPHR ) ) , Innov ) )

       
   #If Parameter estimation is not activated we keep the parameters as in the first analysis cycle.  
   PA[:,:,:,it]=PA[:,:,:,0]


   

   #print('Data assimilation took ', time.time()-start,'seconds.')

print('Data assimilation took ', time.time()-start_cycle,'seconds.')
#=================================================================
#  DIAGNOSTICS  : 
#================================================================= 

SpinUp=200 #Number of assimilation cycles that will be conisdered as spin up 


XAMean=np.mean(XA,axis=1)
XFMean=np.mean(XF,axis=1)

FMean =np.mean(F,axis=1)

XASRmse=np.sqrt( np.mean( np.power( XAMean[:,SpinUp:DALength] - XNature[:,0,SpinUp:DALength] , 2 ) , axis=1 ) )
XFSRmse=np.sqrt( np.mean( np.power( XFMean[:,SpinUp:DALength] - XNature[:,0,SpinUp:DALength] , 2 ) , axis=1 ) )

XATRmse=np.sqrt( np.mean( np.power( XAMean - XNature[:,0,0:DALength] , 2 ) , axis=0 ) )
XFTRmse=np.sqrt( np.mean( np.power( XFMean - XNature[:,0,0:DALength] , 2 ) , axis=0 ) )

XASBias=np.mean( XAMean[:,SpinUp:DALength] - XNature[:,0,SpinUp:DALength]  , axis=1 ) 
XFSBias=np.mean( XFMean[:,SpinUp:DALength] - XNature[:,0,SpinUp:DALength]  , axis=1 ) 

XATBias=np.mean(  XAMean - XNature[:,0,0:DALength]  , axis=0 ) 
XFTBias=np.mean(  XFMean - XNature[:,0,0:DALength]  , axis=0 ) 

FSRmse=np.sqrt( np.mean( np.power( FMean[:,SpinUp:DALength] - FNature[:,0,SpinUp:DALength] , 2 ) , axis=1 ) )
FTRmse=np.sqrt( np.mean( np.power( FMean - FNature[:,0,0:DALength] , 2 ) , axis=0 ) )

FSBias=np.mean( FMean[:,SpinUp:DALength] - FNature[:,0,SpinUp:DALength]  , axis=1 ) 
FTBias=np.mean( FMean - FNature[:,0,0:DALength]  , axis=0 ) 


#=================================================================
#  SAVE OUTPUT  : 
#================================================================= 


if GeneralConf['RunSave']   :
    filename= GeneralConf['DataPath'] + '/' + GeneralConf['OutFile'] 
    print('Saving the output to ' + filename  )
    start = time.time()

    
    if not os.path.exists( GeneralConf['DataPath'] + '/' )  :
      os.makedirs(  GeneralConf['DataPath'] + '/'  )

    #Save Nature run output
    np.savez( filename
                     ,   XA=XA                 
                     ,   PA=PA   
                     ,   F =F
                     ,   XAMean=XAMean         , XFMean=XFMean
                     ,   FMean=FMean           
                     ,   XASRmse=XASRmse       , XFSRmse=XFSRmse
                     ,   XATRmse=XATRmse       , XFTRmse=XFTRmse 
                     ,   XASBias=XASBias       , XFSBias=XFSBias
                     ,   XATBias=XATBias       , XFTBias=XFTBias
                     ,   ModelConf=ModelConf   , DAConf=DAConf
                     ,   GeneralConf=GeneralConf )

#=================================================================
#  PLOT OUTPUT  : 
#=================================================================

#Plot analysis and forecast RMSE and spread.
if GeneralConf['RunPlotState']   :

   if not os.path.exists( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] )  :
      os.makedirs(  GeneralConf['FigPath'] + '/' + GeneralConf['ExpName']  )


   #ANALYSIS-FORECAST RMSE AND SPREAD VS TIME

   plt.figure()

   plt.subplot(2,1,1)
   plt.title('Spatially averaged RMSE and spread')
   rmse, =plt.plot(XATRmse,'b-')
   #sprd, =plt.plot(np.mean(XASpread,axis=0),'r-')
   plt.legend([rmse],['RMSE anl'])
   plt.ylabel('RMSE')
   plt.ylim( (0,XFTRmse[SpinUp:].max()+0.5) )
   plt.grid(True)

   plt.subplot(2,1,2)

   rmse, =plt.plot(XFTRmse,'b-')
   #sprd, =plt.plot(np.mean(XFSpread,axis=0),'r-')
   plt.legend([rmse],['RMSE fcst'])
   plt.ylim( (0,XFTRmse[SpinUp:].max()+0.5) )

   plt.xlabel('Time (cycles)')
   plt.ylabel('RMSE')
   plt.grid(True)

   plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/RMSET_LETKF_run.png', facecolor='w', format='png' )
   plt.show(block=False)
   #plt.close()

   #ANALYSIS-FORECAST AVERAGED RMSE AND SPREAD

   plt.figure()

   plt.subplot(2,1,1)
   plt.title('Time averaged RMSE')
   rmse, =plt.plot(XASRmse,'b-')
   #sprd, =plt.plot(np.mean(XASpread[:,SpinUp:DALength],axis=1),'r-')
   plt.legend([rmse],['RMSE anl'])
   plt.ylim( (0,XFSRmse.max()+0.5) )
   plt.ylabel('RMSE - SPRD')
   plt.grid(True)

   plt.subplot(2,1,2)

   rmse, =plt.plot(XFSRmse,'b-')
   #sprd, =plt.plot(np.mean(XFSpread[:,SpinUp:DALength],axis=1),'r-')
   plt.legend([rmse],['RMSE fcst'])
   plt.ylim( (0,XFSRmse.max()+0.5) )
   plt.grid(True)

   plt.xlabel('Grid point')
   plt.ylabel('RMSE')

   plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/RMSES_LETKF_run.png', facecolor='w', format='png' )
   plt.show(block=False)
   #plt.close()

   #ANALYSIS-FORECAST BIAS

   plt.figure()

   plt.subplot(2,1,1)
   plt.title('Spatially averaged bias')
   bias, =plt.plot(XATBias,'b-')
   plt.legend([bias],['BIAS anl'])
   plt.ylabel('BIAS')
   plt.ylim( (-0.5,0.5) )
   plt.grid(True)

   plt.subplot(2,1,2)

   bias, =plt.plot(XFTBias,'b-')
   plt.legend([bias],['BIAS fcst'])
   plt.ylim( (-0.5,0.5) )

   plt.xlabel('Time (cycles)')
   plt.ylabel('BIAS')
   plt.grid(True)

   #Time series of spatially averaged parameters
   plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/BIAST_LETKF_run.png', facecolor='w', format='png' )
   plt.show(block=False)
   #plt.close()

   #ANALYSIS-FORECAST AVERAGED BIAS

   plt.figure()

   plt.subplot(2,1,1)
   plt.title('Time averaged bias')
   bias, =plt.plot(XASBias,'b-')
   plt.legend([bias],['BIAS anl'])
   plt.ylim( XFSBias.min()-0.5, XFSBias.max()+0.5 )
   plt.ylabel('BIAS')
   plt.grid(True)

   plt.subplot(2,1,2)

   bias, =plt.plot(XFSBias,'b-')
   plt.legend([bias],['BIAS fcst'])
   plt.ylim( XFSBias.min()-0.5, XFSBias.max()+0.5 )
   plt.grid(True)

   plt.xlabel('Grid point')
   plt.ylabel('BIAS')

   plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/BIASS_LETKF_run.png', facecolor='w', format='png' )
   plt.show(block=False)
   #plt.close()

   #State errors
   plt.figure()
   plt.pcolor( XAMean[:,:] - XNature[:,0,0:DALength] , vmin=-2 , vmax=2 )
   plt.xlabel('Time')
   plt.ylabel('Grid Point')
   plt.title('X error')
   plt.colorbar()
   plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/XError.png', facecolor='w', format='png' )
   plt.show(block=False)
   #plt.close()

   #Estimated value, RMSE and SPREAD for the PARAMETERS
   

    



