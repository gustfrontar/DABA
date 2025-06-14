# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:36:06 2017

@author: RISDA 2018
"""

#Run a LETKF experiment using the observations created by the script run_nature.py

import sys
sys.path.append('../model/')
sys.path.append('../data_assimilation/')

from model  import lorenznsto       as model          #Import the model (fortran routines)
from obsope import common_obs       as hoperator      #Import the observation operator (fortran routines)
from da     import common_da_tools  as das            #Import the data assimilation routines (fortran routines)

import matplotlib.pyplot as plt
import numpy as np
import time
import letkf_conf_ParameterEstimationGlobal as conf                #Load the experiment configuration
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

#=================================================================
# INITIALIZATION : 
#=================================================================

#We set the length of the experiment according to the length of the 
#observation array.
DALength = int( max( ObsLoc[:,1] ) / DAConf['Freq'] ) 

#DALength = 3

#Get the number of parameters
NCoef=ModelConf['Coef'].size
#Get the size of the state vector
Nx=ModelConf['nx']
#Get the number of ensembles
NEns=DAConf['NEns']

#Memory allocation and variable definition.

XA=np.zeros([Nx,NEns,DALength])                         #Analisis ensemble
XF=np.zeros([Nx,NEns,DALength])                         #Forecast ensemble
PA=np.zeros([NCoef,NEns,DALength])                      #Estimated parameters

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
CRF=np.zeros([NCoef,NEns])
RF =np.zeros([Nx,NEns])

C0=np.zeros((NCoef,Nx,NEns))

#Generate a random initial conditions and initialize deterministic parameters
for ie in range(0,NEns)  :

   XA[:,ie,0]=ModelConf['Coef'][0]/2 + DAConf['InitialXSigma'] * np.random.normal( size=Nx )

   for ic in range(0,NCoef) :
      PA[ic,ie,0]=ModelConf['Coef'][ic] + DAConf['InitialPSigma'][ic] * np.random.normal( size=1 )
      

#=================================================================
#  MAIN DATA ASSIMILATION LOOP : 
#=================================================================

for it in range( 1 , DALength  )         :

   print('Data assimilation cycle # ',str(it) )

   #=================================================================
   #  ENSEMBLE FORECAST  : 
   #=================================================================   

   #Run the ensemble forecast
   print('Runing the ensemble')

   start = time.time()
   ntout=int( DAConf['Freq'] / DAConf['TSFreq'] ) + 1  #Output the state every ObsFreq time steps.
   
   #Expand the parameters from global constants to local values.
   C=np.zeros([NCoef,Nx,NEns])
   for ic in range(0,NCoef) :
       for ie in range(0,NEns) :
         C[ic,:,ie]=PA[ic,ie,it-1]

   [ XFtmp , DF , RFtmp , CRFtmp, CFtmp ]=model.tinteg_rk4( nens=NEns  , nt=DAConf['Freq'] ,  ntout=ntout ,
                                           x0=XA[:,:,it-1]     , rf0=RF , phi=XPhi , sigma=XSigma,
                                           c0=C  , crf0=CRF    , cphi=CPhi , csigma=CSigma,
                                           nx=Nx, ncoef=NCoef  , dt=ModelConf['dt'] )
   

   XF[:,:,it]=XFtmp[:,:,-1]             #Store the state variables ensemble at the end of the window.

   PF=PA[:,:,it-1]                      #A persistance model is used for the parameters.

   RF=RFtmp[:,:,-1]                     #Store the random forcing state for the state variables.
   CRF=CRFtmp[:,:,-1]                   #Store the random forcing state for the parameters.

   print('Ensemble forecast took ', time.time()-start, 'seconds.')

   #=================================================================
   #  GET THE OBSERVATIONS WITHIN THE TIME WINDOW  : 
   #=================================================================

   print('Observation selection')
   start = time.time()

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
 
   print('Observation selection took ', time.time()-start, 'seconds.')

   #=================================================================
   #  OBSERVATION OPERATOR  : 
   #================================================================= 

   #Apply h operator and transform from model space to observation space. 
   #This opearation is performed for all the observations within the window.
   print('Observation operator')
   start = time.time()

   #Set the time coordinate corresponding to the model output.
   TLoc=np.arange(da_window_start , da_window_end + DAConf['TSFreq'] , DAConf['TSFreq'] )

   #Call the observation operator and transform the ensemble from the state space 
   #to the observation space. 
   [YF , YFmask] = hoperator.model_to_obs(  nx=Nx , no=NObsW , nt=ntout , nens=NEns ,
                                 obsloc=ObsLocW , x=XFtmp , obstype=ObsTypeW ,
                                 xloc=ModelConf['XLoc'] , tloc= TLoc )

   print('Observation operator took ', time.time()-start, 'seconds.')

   #=================================================================
   #  LETKF DA  : 
   #================================================================= 

   print('Data assimilation')

   #STATE VARIABLES ESTIMATION:
  
   start = time.time()

   XA[:,:,it] =np.squeeze( das.da_letkf( nx=Nx , nt=1 , no=NObsW , nens=NEns ,  xloc=ModelConf['XLoc']               ,
                              tloc=da_window_end    , nvar=1                        , xfens=XF[:,:,it]               ,
                              obs=YObsW             , obsloc=ObsLocW                , ofens=YF                       ,
                              rdiag=ObsErrorW       , loc_scale=DAConf['LocScales'] , inf_coefs=DAConf['InfCoefs']   ,
                              update_smooth_coef=0.0 )  )
   
   
   #GLOBAL PARAMETER ESTIMATION (Note that ETKF is used in this case)
   
   PA[:,:,it] =np.squeeze( das.da_etkf( no=NObsW , nens=NEns , nvar=1 , xfens=PF ,
                                        obs=YObsW, ofens=YF  , rdiag=ObsErrorW   ,
                                        inf_coefs=DAConf['InfCoefsP'] ) )



   print('Data assimilation took ', time.time()-start,'seconds.')


#=================================================================
#  DIAGNOSTICS  : 
#================================================================= 

SpinUp=200 #Number of assimilation cycles that will be conisdered as spin up 

XASpread=np.std(XA,axis=1)
XFSpread=np.std(XF,axis=1)

XAMean=np.mean(XA,axis=1)
XFMean=np.mean(XF,axis=1)

XASRmse=np.sqrt( np.mean( np.power( XAMean[:,SpinUp:DALength] - XNature[:,0,SpinUp:DALength] , 2 ) , axis=1 ) )
XFSRmse=np.sqrt( np.mean( np.power( XFMean[:,SpinUp:DALength] - XNature[:,0,SpinUp:DALength] , 2 ) , axis=1 ) )

XATRmse=np.sqrt( np.mean( np.power( XAMean - XNature[:,0,:DALength] , 2 ) , axis=0 ) )
XFTRmse=np.sqrt( np.mean( np.power( XFMean - XNature[:,0,:DALength] , 2 ) , axis=0 ) )

XASBias=np.mean( XAMean[:,SpinUp:DALength] - XNature[:,0,SpinUp:DALength]  , axis=1 ) 
XFSBias=np.mean( XFMean[:,SpinUp:DALength] - XNature[:,0,SpinUp:DALength]  , axis=1 ) 

XATBias=np.mean(  XAMean - XNature[:,0,:DALength]  , axis=0 ) 
XFTBias=np.mean(  XFMean - XNature[:,0,:DALength]  , axis=0 ) 

#Additional computations for the parameter

PAMean=np.mean(PA,axis=1)
PASpread=np.std(PA,axis=1)

PAPRmse=np.sqrt( np.mean( np.power( PAMean[:,SpinUp:DALength] - np.mean( CNature[:,:,0,SpinUp:DALength] , axis=1 ) , 2 ) , axis=1 ) )
PAPBias=np.mean(  PAMean[:,SpinUp:DALength] - np.mean( CNature[:,:,0,SpinUp:DALength] , axis=1 ) , axis=1 ) 

#=================================================================
#  SAVE OUTPUT  : 
#================================================================= 

print('Saving the output to ' +  GeneralConf['DataPath'] + '/' + GeneralConf['ExpName'] + '/' + GeneralConf['LETKFFile'] )
start = time.time()

if GeneralConf['RunSave']   :
   if not os.path.exists( GeneralConf['DataPath'] + '/' + GeneralConf['ExpName'] )  :
      os.makedirs(  GeneralConf['DataPath'] + '/' + GeneralConf['ExpName']  )

   #Save Nature run output
   np.savez( GeneralConf['DataPath'] + '/' + GeneralConf['ExpName'] + '/' + GeneralConf['LETKFFile']
                                                        ,   XA=XA                 , XF=XF
                                                        ,   XAMean=XAMean         , XFMean=XFMean
                                                        ,   XASpread=XASpread     , XFSpread=XFSpread
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
if GeneralConf['RunPlot']   :

   if not os.path.exists( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] )  :
      os.makedirs(  GeneralConf['FigPath'] + '/' + GeneralConf['ExpName']  )


   #ANALYSIS-FORECAST RMSE AND SPREAD VS TIME

   plt.figure()

   plt.subplot(2,1,1)

   rmse, =plt.plot(XATRmse,'b-')
   sprd, =plt.plot(np.mean(XASpread,axis=0),'r-')
   plt.legend([rmse,sprd],['RMSE anl','SPRD anl'])
   plt.ylabel('RMSE - SPRD')
   plt.ylim( (0,1) )
   plt.grid(True)

   plt.subplot(2,1,2)

   rmse, =plt.plot(XFTRmse,'b-')
   sprd, =plt.plot(np.mean(XFSpread,axis=0),'r-')
   plt.legend([rmse,sprd],['RMSE fcst','SPRD fcst'])
   plt.ylim( (0,1) )

   plt.xlabel('Time (cycles)')
   plt.ylabel('RMSE - SPRD')
   plt.grid(True)

   plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/RMSET_LETKF_run.png', facecolor='w', format='png' )
   plt.show()
   #plt.close()

   #ANALYSIS-FORECAST AVERAGED RMSE AND SPREAD

   plt.figure()

   plt.subplot(2,1,1)

   rmse, =plt.plot(XASRmse,'b-')
   sprd, =plt.plot(np.mean(XASpread[:,SpinUp:DALength],axis=1),'r-')
   plt.legend([rmse,sprd],['RMSE anl','SPRD anl'])
   plt.ylim( (0,1) )
   plt.grid(True)

   plt.subplot(2,1,2)

   rmse, =plt.plot(XFSRmse,'b-')
   sprd, =plt.plot(np.mean(XFSpread[:,SpinUp:DALength],axis=1),'r-')
   plt.legend([rmse,sprd],['RMSE fcst','SPRD fcst'])
   plt.ylim( (0,1) )
   plt.grid(True)

   plt.xlabel('X')
   plt.ylabel('RMSE - SPRD')

   plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/RMSES_LETKF_run.png', facecolor='w', format='png' )
   plt.show()
   #plt.close()

   #ANALYSIS-FORECAST BIAS

   plt.figure()

   plt.subplot(2,1,1)

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

   plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/BIAST_LETKF_run.png', facecolor='w', format='png' )
   plt.show()
   #plt.close()

   #ANALYSIS-FORECAST AVERAGED BIAS

   plt.figure()

   plt.subplot(2,1,1)

   bias, =plt.plot(XASBias,'b-')
   plt.legend([bias],['BIAS anl'])
   plt.ylim( (-0.5,0.5) )
   plt.grid(True)

   plt.subplot(2,1,2)

   bias, =plt.plot(XFSBias,'b-')
   plt.legend([bias],['BIAS fcst'])
   plt.ylim( (-0.5,0.5) )
   plt.grid(True)

   plt.xlabel('X')
   plt.ylabel('BIAS')

   plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/BIASS_LETKF_run.png', facecolor='w', format='png' )
   plt.show()
   #plt.close()















