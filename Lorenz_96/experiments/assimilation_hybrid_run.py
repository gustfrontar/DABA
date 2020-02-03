# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:36:06 2017
@author: RISDA 2018
"""

#Run a Hybrid ETPF-LETKF experiment using the observations created by the script run_nature.py
#Also a tempered ETPF or LETKF can be run using this script.

import sys
sys.path.append('../model/')
sys.path.append('../data_assimilation/')

from model  import lorenzn          as model          #Import the model (fortran routines)
from obsope import common_obs       as hoperator      #Import the observation operator (fortran routines)
from da     import common_da_tools  as das            #Import the data assimilation routines (fortran routines)

import matplotlib.pyplot as plt
import numpy as np
import time
import assimilation_conf_HybridPerfectModel_R1_Den1_Freq16_Hlinear as conf         #Load the experiment configuration
from scipy import stats
import os

np.random.seed(20)

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

if DAConf['ExpLength'] == None :
   DALength = int( max( ObsLoc[:,1] ) / DAConf['Freq'] )
else:
   DALength = DAConf['ExpLength']
   XNature = XNature[:,:,0:DALength+1]
   CNature = CNature[:,:,:,0:DALength+1] 
   FNature = FNature[:,:,0:DALength+1]
   
   
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
   RandInd1=(np.round(np.random.rand(1)*DALength)).astype(int)
   RandInd2=(np.round(np.random.rand(1)*DALength)).astype(int)

   #XA[:,ie,0]=ModelConf['Coef'][0]/2 + DAConf['InitialXSigma'] * np.random.normal( size=Nx )
   #Reemplazo el perturbado totalmente random por un perturbado mas inteligente.
   XA[:,ie,0]=ModelConf['Coef'][0]/2 + np.squeeze( DAConf['InitialXSigma'] * ( XNature[:,0,RandInd1] - XNature[:,0,RandInd2] ) )
     
    
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
   if np.mod(it,100) == 0  :
      print('Data assimilation cycle # ',str(it) )

   #=================================================================
   #  ADD ADDITIVE ENSEMBLE PERTURBATIONS  : 
   #=================================================================
   #Additive perturbations will be generated as scaled random
   #differences of nature run states.
   if DAConf['InfCoefs'][4] > 0.0 :
      #Get random index to generate additive perturbations
      RandInd1=(np.round(np.random.rand(NEns)*DALength)).astype(int)
      RandInd2=(np.round(np.random.rand(NEns)*DALength)).astype(int)
   
      AddInfPert = np.squeeze( XNature[:,0,RandInd1] - XNature[:,0,RandInd2] ) * DAConf['InfCoefs'][4]

      #Shift perturbations to obtain zero-mean perturbations.   
      AddInfPertMean = np.mean( AddInfPert , 1)
      for ie in range(NEns)  :
         AddInfPert[:,ie] = AddInfPert[:,ie] - AddInfPertMean
      
      XA[:,:,it-1] = XA[:,:,it-1] + AddInfPert 
      
   #=================================================================
   #  ENSEMBLE FORECAST  : 
   #=================================================================   

   #Run the ensemble forecast
   #print('Runing the ensemble')

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
 
   #=================================================================
   #  HYBRID-TEMPERED DA  : 
   #================================================================= 

   gamma = 1.0/DAConf['NTemp']

   stateens = np.copy(XF[:,:,it])
   
   #Perform initial iterations using ETKF this helps to speed up convergence.
   if it < DAConf['NKalmanSpinUp']  :
       BridgeParam = 0.0  #Force pure Kalman step.
   else                             :
       BridgeParam = DAConf['BridgeParam']
       
   
   
   for itemp in range( DAConf['NTemp'] ) :
       
       
      #=================================================================
      #  OBSERVATION OPERATOR  : 
      #================================================================= 

      #Apply h operator and transform from model space to observation space. 
      #This opearation is performed only at the end of the window.

      #Set the time coordinate corresponding to the model output.
      TLoc= da_window_end #We are assuming that all observations are valid at the end of the assimilaation window.
      #Call the observation operator and transform the ensemble from the state space 
      #to the observation space. 
      [YF , YFmask] = hoperator.model_to_obs(  nx=Nx , no=NObsW , nt=1 , nens=NEns ,
                             obsloc=ObsLocW , x=stateens , obstype=ObsTypeW ,
                             xloc=ModelConf['XLoc'] , tloc= TLoc )
       
      #=================================================================
      #  LETKF STEP  : 
      #=================================================================
      
      if BridgeParam < 1.0 :

         local_obs_error = ObsErrorW * DAConf['NTemp'] / ( 1.0 - BridgeParam ) 
         stateens = das.da_letkf( nx=Nx , nt=1 , no=NObsW , nens=NEns ,  xloc=ModelConf['XLoc']               ,
                              tloc=da_window_end    , nvar=1                        , xfens=stateens               ,
                              obs=YObsW             , obsloc=ObsLocW                , ofens=YF                       ,
                              rdiag=local_obs_error , loc_scale=DAConf['LocScalesLETKF'] , inf_coefs=DAConf['InfCoefs']   ,
                              update_smooth_coef=0.0 )[:,:,0,0]

      #=================================================================
      #  ETPF STEP  : 
      #=================================================================
         
      if BridgeParam > 0.0 :

          local_obs_error = ObsErrorW * DAConf['NTemp'] / ( BridgeParam )
          [tmp_ens , wa]= das.da_letpf( nx=Nx , nt=1 , no=NObsW , nens=NEns ,  xloc=ModelConf['XLoc']               ,
                                       tloc=da_window_end    , nvar=1                        , xfens=stateens               , 
                                       obs=YObsW             , obsloc=ObsLocW                , ofens=YF                     ,
                                       rdiag=local_obs_error , loc_scale=DAConf['LocScalesLETPF'] , rejuv_param=DAConf['RejuvParam']  )
          stateens = tmp_ens[:,:,0,0]
       


   #print( np.mean( np.std(stateens,1) ) / np.mean( np.std(XF[:,:,it] ,1 ) ) )                     
                             
  
   XA[:,:,it] = np.copy( stateens )
   
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

print('Data assimilation took ', time.time()-start_cycle,'seconds.')
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

print(' Analysis RMSE ',np.mean(XASRmse),' Analysis SPREAD ',np.mean(XASpread))

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
                     ,   XASpread=XASpread     , XFSpread=XFSpread
                     ,   PAMean=PAMean         , PASpread=PASpread
                     ,   FMean=FMean           , FSpread=FSpread
                     ,   XASRmse=XASRmse       , XFSRmse=XFSRmse
                     ,   XATRmse=XATRmse       , XFTRmse=XFTRmse 
                     ,   XASBias=XASBias       , XFSBias=XFSBias
                     ,   XATBias=XATBias       , XFTBias=XFTBias
                     ,   PASRmse=PASRmse       , PASBias=PASBias
                     ,   PATRmse=PATRmse       , PATBias=PATBias
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
   sprd, =plt.plot(np.mean(XASpread,axis=0),'r-')
   plt.legend([rmse,sprd],['RMSE anl','SPRD anl'])
   plt.ylabel('RMSE - SPRD')
   plt.ylim( (0,XFTRmse[SpinUp:].max()+0.5) )
   plt.grid(True)

   plt.subplot(2,1,2)

   rmse, =plt.plot(XFTRmse,'b-')
   sprd, =plt.plot(np.mean(XFSpread,axis=0),'r-')
   plt.legend([rmse,sprd],['RMSE fcst','SPRD fcst'])
   plt.ylim( (0,XFTRmse[SpinUp:].max()+0.5) )

   plt.xlabel('Time (cycles)')
   plt.ylabel('RMSE - SPRD')
   plt.grid(True)

   plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/RMSET_LETKF_run.png', facecolor='w', format='png' )
   plt.show(block=False)
   #plt.close()

   #ANALYSIS-FORECAST AVERAGED RMSE AND SPREAD

   plt.figure()

   plt.subplot(2,1,1)
   plt.title('Time averaged RMSE and spread')
   rmse, =plt.plot(XASRmse,'b-')
   sprd, =plt.plot(np.mean(XASpread[:,SpinUp:DALength],axis=1),'r-')
   plt.legend([rmse,sprd],['RMSE anl','SPRD anl'])
   plt.ylim( (0,XFSRmse.max()+0.5) )
   plt.ylabel('RMSE - SPRD')
   plt.grid(True)

   plt.subplot(2,1,2)

   rmse, =plt.plot(XFSRmse,'b-')
   sprd, =plt.plot(np.mean(XFSpread[:,SpinUp:DALength],axis=1),'r-')
   plt.legend([rmse,sprd],['RMSE fcst','SPRD fcst'])
   plt.ylim( (0,XFSRmse.max()+0.5) )
   plt.grid(True)

   plt.xlabel('Grid point')
   plt.ylabel('RMSE - SPRD')

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
   
for ic in range(0,NCoef)  :
   if GeneralConf['RunPlotParameters'] & (DAConf['InitialPSigma'][ic] > 0)  :

      CString=str(ic)
      
      #Time series of spatially averaged parameters
      plt.figure()
      plt.plot( np.mean( PAMean[:,ic,0:DALength], axis=0) ,'b-')
      plt.plot( np.mean( CNature[:,0,ic,0:DALength] , axis=0) , 'k--')
      plt.xlabel('Time')
      plt.ylim(PAMean[:,ic,0:DALength].min()-0.25,PAMean[:,ic,0:DALength].max()+0.25 )
      plt.ylabel('Parameter value')
      plt.title('Estimated parameter ' + CString + ' value')
      plt.grid(True)
      plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/PEstimated' + CString + 'vsTime.png', facecolor='w', format='png' )
      plt.show(block=False)
      #plt.close()
     
      #Spatiall distribution of time averaged parameters
      plt.figure()
      meanPA=np.mean( PAMean[:,ic,SpinUp:DALength], axis=1)
      meanPNature=np.mean( CNature[:,0,ic,SpinUp:DALength] , axis=1)
      plt.plot( meanPA ,'b-')
      plt.plot( meanPNature , 'k--')
      plt.xlabel('Grid point')
      plt.ylim(meanPA.min()-0.25,meanPA.max()+0.25 )
      plt.ylabel('Parameter value')
      plt.title('Estimated parameter ' + CString + ' value')
      plt.grid(True)
      plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/PEstimated' + CString + 'vsGridPoint.png', facecolor='w', format='png' )
      plt.show(block=False)
      #plt.close()
     
 
       
      plt.figure()
      rmse, =plt.plot(PATRmse[ic,:],'b-')
      sprd, =plt.plot(np.mean(PASpread[:,ic,:],axis=0),'r-')
      plt.legend([rmse,sprd],['RMSE anl','SPRD anl'])
      plt.ylabel('RMSE - SPRD')
      plt.xlabel('Time')
      plt.ylim( (0,PATRmse[ic,SpinUp:].max()+0.5) )
      plt.grid(True)
      plt.title('Parameter ' + CString)
      plt.show(block=False)
      plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/RMSEP_Anl_LETKF_run.png', facecolor='w', format='png' )
      #plt.close()

      plt.figure()
      rmse, =plt.plot(PASRmse[:,ic],'b-')
      sprd, =plt.plot(np.mean(PASpread[:,ic,SpinUp:DALength],axis=1),'r-')
      plt.legend([rmse,sprd],['RMSE anl','SPRD anl'])
      plt.ylim( (0,PASRmse[:,ic].max()+0.5) )
      plt.xlabel('Grid Point')
      plt.ylabel('RMSE')
      plt.grid(True)
     
      plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/RMSEP_For_LETKF_run.png', facecolor='w', format='png' )
      plt.show(block=False)
      #plt.close()
     
      plt.figure()
      plt.pcolor( PAMean[:,ic,:] - CNature[:,0,ic,0:DALength])
      plt.xlabel('Time')
      plt.ylabel('Grid Point')
      plt.title('Parameter ' + CString + ' error')
      plt.colorbar()
      plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/P' + CString + 'Error.png', facecolor='w', format='png' )
      plt.show(block=False)
      #plt.close()
     
      #Space-Time estimated parameter
      plt.figure()
      plt.pcolor( PAMean[:,ic,0:DALength] )
      plt.xlabel('Time')
      plt.ylabel('Grid Point')
      plt.title('Estimated Parameter ' + CString )
      plt.colorbar()
      plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/PEstimated' + CString + '_SpaceTime.png', facecolor='w', format='png' )
      plt.show(block=False)
      #plt.close()
      
      #Space-Time true parameter
      plt.figure()
      plt.pcolor( CNature[:,0,ic,0:DALength] )
      plt.xlabel('Time')
      plt.ylabel('Grid Point')
      plt.title('True Parameter ' + CString )
      plt.colorbar()
      plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/PTrue' + CString + '_SpaceTime.png', facecolor='w', format='png' )
      plt.show(block=False)
      #plt.close() 
   
    
if GeneralConf['RunPlotForcing']     :
   
   #Forcing errors
   plt.figure()
   plt.pcolor( FMean[:,:] - FNature[:,0,0:DALength] , vmin=-4 , vmax=4 )
   plt.xlabel('Time')
   plt.ylabel('Grid Point')
   plt.title('F error')
   plt.colorbar()
   plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/FError.png', facecolor='w', format='png' )
   plt.show(block=False)
   #plt.close()
   

   #Plot total forcing as a function of X
   plt.figure()
   tmpX=np.reshape( XAMean[:,SpinUp:DALength] , [ FMean[:,SpinUp:DALength].shape[0] * FMean[:,SpinUp:DALength].shape[1] ] ) #A vector containing all X
   tmpF=np.reshape( PAMean[:,0,SpinUp:DALength] , [ FMean[:,SpinUp:DALength].shape[0] * FMean[:,SpinUp:DALength].shape[1] ] ) #A vector containing all Forcings
   plt.scatter(tmpX,tmpF,s=0.5,c="g", alpha=0.5, marker='o')
   [a,b,r,p,std_err]= stats.linregress(tmpX,tmpF)
   
   LinearX=np.arange(round(tmpX.min())-1,round(tmpX.max())+1)
   LinearF=b + LinearX * a
   reg, =plt.plot(LinearX,LinearF,'--k')
   plt.legend([reg],['Linear Reg a=' + str(a) + ' b=' + str(b) ])
   plt.title('Forcing vs X')
   plt.xlabel('X')
   plt.ylabel('Estimated forcing')
   plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/Forcing_vs_X.png', facecolor='w', format='png' )
   plt.show(block=False)
   #plt.close()

   plt.figure()
   truef,=plt.plot(FNature[0,0,SpinUp:])
   estf,=plt.plot(PAMean[0,0,SpinUp:])
   plt.legend([truef,estf],['True','Est.'])
   plt.title('True and estimated forcing at 1st grid point')
   plt.xlabel('Time')
   plt.ylabel('Forcing')
   plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/Forcing_at_first_gridpoint.png', facecolor='w', format='png' )
   plt.show(block=False)
   #plt.close()
   
   #FORECAST FORCING RMSE and BIAS

   plt.figure()

   plt.subplot(2,1,1)

   rmse, =plt.plot(FTRmse,'b-')
   plt.legend([rmse],['RMSE Forcing'])
   plt.ylim( (0,FTRmse[SpinUp:].max()+1) )
   plt.grid(True)
   plt.ylabel('RMSE')

   plt.subplot(2,1,2)

   bias, =plt.plot(FTBias,'b-')
   plt.legend([bias],['BIAS Forcing'])
   plt.ylim( (FTBias[SpinUp:].min()-1,FTBias[SpinUp:].max()+1) )
   plt.grid(True)

   plt.xlabel('Time')
   plt.ylabel('Bias')

   plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/ForcingError_LETKF_run.png', facecolor='w', format='png' )
   plt.show(block=False)
   #plt.close()


plt.show()

