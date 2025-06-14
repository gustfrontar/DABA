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
import assimilation_conf_ImperfectModel as conf         #Load the experiment configuration
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
  #print('')
  #print('This is a TWIN experiment')
  #print('')
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
NEns=DAConf['NEns']

#Memory allocation and variable definition.

XA=np.zeros([Nx,NEns,DALength])                         #Analisis ensemble
XF=np.zeros([Nx,NEns,DALength])                         #Forecast ensemble
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

  
   #print('Data assimilation cycle # ',str(it) )

   #=================================================================
   #  ENSEMBLE FORECAST  : 
   #=================================================================   

   #Run the ensemble forecast
   #print('Runing the ensemble')

   #start = time.time()
   ntout=int( DAConf['Freq'] / DAConf['TSFreq'] ) + 1  #Output the state every ObsFreq time steps.
   
   [ XFtmp , XSStmp , DFtmp , RFtmp , SSFtmp , CRFtmp, CFtmp ]=model.tinteg_rk4( nens=NEns  , nt=DAConf['Freq'] ,  ntout=ntout ,
                                           x0=XA[:,:,it-1]     , xss0=XSS , rf0=RF    , phi=XPhi     , sigma=XSigma,
                                           c0=PA[:,:,:,it-1]   , crf0=CRF             , cphi=CPhi    , csigma=CSigma, param=ModelConf['TwoScaleParameters'] , 
                                           nx=Nx,  nxss=NxSS   , ncoef=NCoef  , dt=ModelConf['dt']   , dtss=ModelConf['dtss'])

   PF[:,:,:,it] = CFtmp[:,:,:,-1]       #Store the parameter at the end of the window. 
   XF[:,:,it]=XFtmp[:,:,-1]             #Store the state variables ensemble at the end of the window.

   F[:,:,it] =DFtmp[:,:,-1]+RFtmp[:,:,-1]+SSFtmp[:,:,-1]  #Store the total forcing 
   
   XSS=XSStmp[:,:,-1]
   CRF=CRFtmp[:,:,-1]
   RF=RFtmp[:,:,-1]
   
   #print('Ensemble forecast took ', time.time()-start, 'seconds.')

   #=================================================================
   #  GET THE OBSERVATIONS WITHIN THE TIME WINDOW  : 
   #=================================================================

   #print('Observation selection')
   #start = time.time()

   da_window_start  = (it -1) * DAConf['Freq']
   da_window_end    = da_window_start + DAConf['Freq']
   da_analysis_time = da_window_end

   #Screen the observations and get only the onew within the da window
   window_mask=np.logical_and( ObsLoc[:,1] > da_window_start , ObsLoc[:,1] <= da_window_end )
 
   ObsLocW=ObsLoc[window_mask,:]                                     #Observation location within the DA window.
   ObsTypeW=ObsType[window_mask]                                     #Observation type within the DA window
   YObsW=YObs[window_mask]                                           #Observations within the DA window
   NObsW=YObsW.size                                                  #Number of observations within the DA window
   ObsErrorW=ObsError[window_mask]                                   #Observation error within the DA window         
 
   #print('Observation selection took ', time.time()-start, 'seconds.')

   #=================================================================
   #  OBSERVATION OPERATOR  : 
   #================================================================= 

   #Apply h operator and transform from model space to observation space. 
   #This opearation is performed for all the observations within the window.
   #print('Observation operator')
   #start = time.time()

   #Set the time coordinate corresponding to the model output.
   TLoc=np.arange(da_window_start , da_window_end + DAConf['TSFreq'] , DAConf['TSFreq'] )

   #Call the observation operator and transform the ensemble from the state space 
   #to the observation space. 
   [YF , YFmask] = hoperator.model_to_obs(  nx=Nx , no=NObsW , nt=ntout , nens=NEns ,
                                 obsloc=ObsLocW , x=XFtmp , obstype=ObsTypeW ,
                                 xloc=ModelConf['XLoc'] , tloc= TLoc )

   #print('Observation operator took ', time.time()-start, 'seconds.')

   #=================================================================
   #  LETKF DA  : 
   #================================================================= 

   #print('Data assimilation')

   #STATE VARIABLES ESTIMATION:
  
   #start = time.time()

   XA[:,:,it] = das.da_letkf( nx=Nx , nt=1 , no=NObsW , nens=NEns ,  xloc=ModelConf['XLoc']               ,
                              tloc=da_window_end    , nvar=1                        , xfens=XF[:,:,it]               ,
                              obs=YObsW             , obsloc=ObsLocW                , ofens=YF                       ,
                              rdiag=ObsErrorW       , loc_scale=DAConf['LocScales'] , inf_coefs=DAConf['InfCoefs']   ,
                              update_smooth_coef=0.0 )[:,:,0,0]
   
   #PARAMETER ESTIMATION
   if DAConf['EstimateParameters']   : 
      
    if DAConf['ParameterLocalizationType'] == 1  :
       #GLOBAL PARAMETER ESTIMATION (Note that ETKF is used in this case)
   
       PA[:,:,:,it] = das.da_etkf( no=NObsW , nens=NEns , nvar=NCoef , xfens=PF[:,:,:,it] ,
                                            obs=YObsW, ofens=YF  , rdiag=ObsErrorW   ,
                                            inf_coefs=DAConf['InfCoefsP'] )[:,:,:,0] 
       
    if DAConf['ParameterLocalizationType'] == 2  :
       #GLOBAL AVERAGED PARAMETER ESTIMATION (Parameters are estiamted locally but the agregated globally)
       #LETKF is used but a global parameter is estimated.
       
       #First estimate a local value for the parameters at each grid point.
       PA[:,:,:,it] = das.da_letkf( nx=Nx , nt=1 , no=NObsW , nens=NEns ,  xloc=ModelConf['XLoc']      ,
                              tloc=da_window_end    , nvar=NCoef                    , xfens=PF[:,:,:,it]             ,
                              obs=YObsW             , obsloc=ObsLocW                , ofens=YF                       ,
                              rdiag=ObsErrorW       , loc_scale=DAConf['LocScalesP'] , inf_coefs=DAConf['InfCoefsP']   ,
                              update_smooth_coef=0.0 )[:,:,:,0]
       
       #Spatially average the estimated parameters so we get the same parameter values
       #at each model grid point.
       for ic in range(0,NCoef)  :
           for ie in range(0,NEns)  :
              PA[:,ie,ic,it]=np.mean( PA[:,ie,ic,it] , axis = 0 )
              
    if DAConf['ParameterLocalizationType'] == 3 :
       #LOCAL PARAMETER ESTIMATION (Parameters are estimated at each model grid point and the forecast uses 
       #the locally estimated parameters)
       #LETKF is used to get the local value of the parameter.
       PA[:,:,:,it] = das.da_letkf( nx=Nx , nt=1 , no=NObsW , nens=NEns ,  xloc=ModelConf['XLoc']      ,
                              tloc=da_window_end    , nvar=NCoef                    , xfens=PF[:,:,:,it]             ,
                              obs=YObsW             , obsloc=ObsLocW                , ofens=YF                       ,
                              rdiag=ObsErrorW       , loc_scale=DAConf['LocScalesP'] , inf_coefs=DAConf['InfCoefsP']   ,
                              update_smooth_coef=0.0 )[:,:,:,0]
       
       
   else :
    #If Parameter estimation is not activated we keep the parameters as in the first analysis cycle.  
    PA[:,:,:,it]=PA[:,:,:,0]


   

   #print('Data assimilation took ', time.time()-start,'seconds.')

#print('Data assimilation took ', time.time()-start_cycle,'seconds.')
#=================================================================
#  DIAGNOSTICS  : 
#================================================================= 

SpinUp=200 #Number of assimilation cycles that will be conisdered as spin up 

XASpread=np.std(XA,axis=1)
XFSpread=np.std(XF,axis=1)

FSpread =np.std(F,axis=1)

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

#Additional computations for the parameter

PAMean=np.mean(PA,axis=1)
PASpread=np.std(PA,axis=1)

PFMean=np.mean(PF,axis=1)
PFSpread=np.std(PF,axis=1)

PASRmse=np.sqrt( np.mean( np.power( PAMean[:,:,SpinUp:DALength] - CNature[:,0,:,SpinUp:DALength] , 2 ) , axis=2 ) )
PFSRmse=np.sqrt( np.mean( np.power( PFMean[:,:,SpinUp:DALength] - CNature[:,0,:,SpinUp:DALength] , 2 ) , axis=2 ) )

PATRmse=np.sqrt( np.mean( np.power( PAMean - CNature[:,0,:,0:DALength] , 2 ) , axis=0 ) )
PFTRmse=np.sqrt( np.mean( np.power( PFMean - CNature[:,0,:,0:DALength] , 2 ) , axis=0 ) )

PASBias=np.mean( PAMean[:,:,SpinUp:DALength] - CNature[:,0,:,SpinUp:DALength] , axis=2 ) 
PFSBias=np.mean( PFMean[:,:,SpinUp:DALength] - CNature[:,0,:,SpinUp:DALength]  , axis=2 ) 

PATBias=np.mean(  PAMean - CNature[:,0,:,0:DALength]  , axis=0 ) 
PFTBias=np.mean(  PFMean - CNature[:,0,:,0:DALength]  , axis=0 )



print('Loc scale ',DAConf['LocScales'][0],'ANA RMSE',np.mean(XASRmse),'F RMSE',np.mean(XFSRmse) )
