# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:36:06 2017

This version implements the two step semi-linear filter.

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
import assimilation_conf_refobs as conf         #Load the experiment configuration
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

NRealizations = 1000

RMSEA_1=np.zeros(NRealizations)
RMSEA_2=np.zeros(NRealizations)
RMSEA_O=np.zeros(NRealizations)
RMSEF=np.zeros(NRealizations)


for ir in range( 0 , NRealizations ) :
 
   #Generate a random initial conditions and initialize deterministic parameters
   for ie in range(0,NEns)  :

      #XA[:,ie,0]=ModelConf['Coef'][0]/2 + DAConf['InitialXSigma'] * np.random.normal( size=Nx )
      XA[:,ie,0]=XNature[:,0,0] + 3.0 * np.random.normal( size=Nx ) + 5.0
           
           
   XA2=np.copy(XA) 
   XAO=np.copy(XA)          
#=================================================================
#  MAIN DATA ASSIMILATION LOOP : 
#=================================================================
   #start_cycle = time.time()

   #for it in range( 1 , 2 ) : #DALength  )         :

  
   print('Data assimilation cycle # ',str(ir) )

   #=================================================================
   #  ENSEMBLE FORECAST  : 
   #=================================================================   

   #Run the ensemble forecast
   #print('Runing the ensemble')

   #start = time.time()
   ntout=int( DAConf['Freq'] / DAConf['TSFreq'] ) + 1  #Output the state every ObsFreq time steps.
   
   #print(XA[:,:,it-1])
   
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

   print('Observation selection')
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
   print('Observation operator')
   #start = time.time()

   #Set the time coordinate corresponding to the model output.
   
   
   TLoc=np.arange(da_window_start , da_window_end + DAConf['TSFreq'] , DAConf['TSFreq'] )

   #Call the observation operator and transform the ensemble from the state space 
   #to the observation space. 
   [YF , YFmask] = hoperator.model_to_obs(  nx=Nx , no=NObsW , nt=ntout , nens=NEns ,
                                 obsloc=ObsLocW , x=XFtmp , obstype=ObsTypeW ,
                                 xloc=ModelConf['XLoc'] , tloc= TLoc )

   #print('Observation operator took ', time.time()-start, 'seconds.')
   #Run observation preprosesor for psedo radar observations. 
   if ObsConf['Type'] == 3  :
       LowerObsLimit=-20.0
       YF[ YF <= LowerObsLimit ]=LowerObsLimit
       YObsW[YObsW <= LowerObsLimit ]= LowerObsLimit
       YFmask[ np.mean(YF,1) == LowerObsLimit ] = LowerObsLimit

   #=================================================================
   #  PF STEP, GENERATE PSEUDO OBSERVATIONS.  
   #================================================================= 

   #Compute importance sampling solution (without resampling)
   #First compute the PF update in the state variables from the "complex" observables.
   #The PF posterior mean or mode in the state variables will be used as pseudo observations 
   #for the EnKF.
   print('Calling PF pseudo obs generator')
   [ XPF_MEAN , XPF_MODE , XPF_STD , WA ] = das.da_lpf( nx=Nx , nt=1 , no=NObsW , nens=NEns ,  xloc=ModelConf['XLoc']               ,
                              tloc=da_window_end    , nvar=1                        , xfens=XF[:,:,it]               ,
                              obs=YObsW             , obsloc=ObsLocW                , ofens=YF                       ,
                              rdiag=ObsErrorW       , loc_scale=np.array([0.1,-1.0]) ) #DAConf['LocScales'] )

   
   #Generate Pseudo observations assuming 
   # 1) Diagonal R
   # 2) Fully observed state.
   # 3) Observation is based on the PF ensemble mean.
   
   #Pseudo observations full.
   
   PseudoNObsW=Nx
   PseudoObsLocW=np.zeros([Nx,2])
   PseudoObsLocW[:,1]=np.copy(TLoc[1])
   PseudoObsLocW[0:,0]=np.arange(0.0,40.0,1.0) + 1.0
   PseudoObsTypeW=np.ones(Nx)
   
   
   #Run the observation operator again, now assuming that the state variables are directly observed.
   print('Running observation operator for pseudo observations')
   [PseudoYF , PseudoYFmask] = hoperator.model_to_obs(  nx=Nx , no=PseudoNObsW , nt=ntout , nens=NEns ,
                                 obsloc=PseudoObsLocW , x=XFtmp , obstype=PseudoObsTypeW ,
                                 xloc=ModelConf['XLoc'] , tloc= TLoc )

   PseudoYObsW= np.copy(XPF_MODE)
   PseudoObsErrorW=np.ones(np.shape(XPF_STD)) * 0.5   #Error of pseudo observations.
   
   
   #Pseudo observations hybrid (observatciones del PF cuando la diferencia es grande)
   index = np.abs( np.squeeze( np.mean(YF,1) ) - np.squeeze(YObsW)  ) > 1.0 
   
   Pseudo2NObsW=Nx
   Pseudo2ObsLocW=np.zeros([Nx,2])
   Pseudo2ObsLocW[:,1]=np.copy(TLoc[1])
   Pseudo2ObsLocW[0:,0]=np.arange(0.0,40.0,1.0) + 1.0
   
   Pseudo2ObsTypeW=np.copy( ObsTypeW )
   Pseudo2ObsTypeW[index]=1.0
   
   Pseudo2YObsW = np.copy(YObsW)
   Pseudo2YObsW[index,0] = PseudoYObsW[index,0,0]
   
   Pseudo2ObsErrorW= np.copy( ObsErrorW )
   Pseudo2ObsErrorW[index,0] = PseudoObsErrorW[index,0,0]
   
   Pseudo2YF = np.copy(YF)
   Pseudo2YF[index,:] = PseudoYF[index,:]
   
   Pseudo2YFmask = np.copy( YFmask )
   Pseudo2YFmask[index] = PseudoYFmask[index]
    
   #=================================================================
   #  LETKF DA  : 
   #================================================================= 

   #print('Data assimilation')

   #STATE VARIABLES ESTIMATION:
  
   #start = time.time()

   XA[:,:,it] = das.da_letkf( nx=Nx , nt=1 , no=PseudoNObsW , nens=NEns ,  xloc=ModelConf['XLoc']               ,
                              tloc=da_window_end    , nvar=1                        , xfens=XF[:,:,it]               ,
                              obs=PseudoYObsW       , obsloc=PseudoObsLocW          , ofens=PseudoYF                 ,
                              rdiag=PseudoObsErrorW , loc_scale=DAConf['LocScales'] , inf_coefs=DAConf['InfCoefs']   ,
                              update_smooth_coef=0.0 )[:,:,0,0]

   XA2[:,:,it] = das.da_letkf( nx=Nx , nt=1 , no=Pseudo2NObsW , nens=NEns ,  xloc=ModelConf['XLoc']               ,
                              tloc=da_window_end    , nvar=1       , xfens=XF[:,:,it]               ,
                              obs=Pseudo2YObsW       , obsloc=Pseudo2ObsLocW     , ofens=Pseudo2YF                 ,
                              rdiag=Pseudo2ObsErrorW , loc_scale=DAConf['LocScales'] , inf_coefs=DAConf['InfCoefs']   ,
                              update_smooth_coef=0.0 )[:,:,0,0]

   XAO[:,:,it] = das.da_letkf( nx=Nx , nt=1 , no=NObsW , nens=NEns ,  xloc=ModelConf['XLoc']               ,
                              tloc=da_window_end    , nvar=1       , xfens=XF[:,:,it]               ,
                              obs=YObsW       , obsloc=ObsLocW     , ofens=YF                 ,
                              rdiag=ObsErrorW , loc_scale=DAConf['LocScales'] , inf_coefs=DAConf['InfCoefs']   ,
                              update_smooth_coef=0.0 )[:,:,0,0]
   
   
   
   #plt.figure()
   #plt.plot(np.squeeze(XPF_MODE),'b');plt.plot(XNature[:,0,it],'k');plt.plot(np.mean(XF[:,:,it],1),'r')

   #tmp= np.argmin( np.squeeze(XNature[:,0,it])-np.squeeze(XPF_MODE) )
#   tmp = np.argmax( np.squeeze(XPF_MODE) )
#   print('index',tmp)
#   print('YO',YObsW[tmp])
#   print('YF',YF[tmp,:])
#   print('XF',XF[tmp,:,it])
#
#   print('PYO',PseudoYObsW[tmp])
#   print('Xt',XNature[tmp,0,it])
#   print('PF_MODE',XPF_MODE[tmp])
#   print('PF_MEAN',XPF_MEAN[tmp])
#   
   #print('RMSE A',np.sqrt( np.mean( np.power( np.squeeze(XNature[:,0,it])-
   #                               np.squeeze(np.mean(XA[:,:,it],1)) , 2 ) ) ) )
   
   #print('RMSE A2',np.sqrt( np.mean( np.power( np.squeeze(XNature[:,0,it])-
   #                               np.squeeze(np.mean(XA2[:,:,it],1)) , 2 ) ) ) )
   
   #print('RMSE F',np.sqrt( np.mean( np.power( np.squeeze(XNature[:,0,it])-
   #                               np.squeeze(np.mean(XF[:,:,it],1)) , 2 ) ) ) )

   RMSEA_1[ir]=np.sqrt( np.mean( np.power( np.squeeze(XNature[:,0,it])-
                                  np.squeeze(np.mean(XA[:,:,it],1)) , 2 ) ) ) 
   
   RMSEA_2[ir]=np.sqrt( np.mean( np.power( np.squeeze(XNature[:,0,it])-
                                  np.squeeze(np.mean(XA2[:,:,it],1)) , 2 ) ) ) 
   
   RMSEA_O[ir]=np.sqrt( np.mean( np.power( np.squeeze(XNature[:,0,it])-
                                  np.squeeze(np.mean(XAO[:,:,it],1)) , 2 ) ) ) 
   
   RMSEF[ir]  =np.sqrt( np.mean( np.power( np.squeeze(XNature[:,0,it])-
                                  np.squeeze(np.mean(XF[:,:,it],1)) , 2 ) ) ) 

   
print('Data assimilation took ', time.time()-start_cycle,'seconds.')


print('RMSEA_1',np.mean(RMSEA_1))
print('RMSEA_2',np.mean(RMSEA_2))
print('RMSEA_O',np.mean(RMSEA_O))
print('RMSEF',np.mean(RMSEF))
