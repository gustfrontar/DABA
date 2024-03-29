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

import assimilation_conf_test as conf         #Load the experiment configuration

import numpy as np
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
DAConf['Freq']=ObsConf['Freq']
DAConf['TSFreq']=ObsConf['Freq']

    
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
NAssimObs=np.zeros(DALength)
    
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
  
   #Reemplazo el perturbado totalmente random por un perturbado mas inteligente.
   XA[:,ie,0]=ModelConf['Coef'][0]/2 + np.squeeze( DAConf['InitialXSigma'] * ( XNature[:,0,RandInd1] - XNature[:,0,RandInd2] ) )
         
        
   for ic in range(0,NCoef) : 
       PA[:,ie,ic,0]=ModelConf['Coef'][ic] + DAConf['InitialPSigma'][ic] * np.random.normal( size=1 )
               
#=================================================================
#  MAIN DATA ASSIMILATION LOOP : 
#=================================================================
    
for it in range( 1 , DALength  )         :
   if np.mod(it,100) == 0  :
      print('Data assimilation cycle # ',str(it) )
    
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
       
   #=================================================================
   #  GET THE OBSERVATIONS WITHIN THE TIME WINDOW  : 
   #=================================================================
    
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
    
   stateens = np.copy(XF[:,:,it])

   #=================================================================
   #  OBSERVATION OPERATOR  : 
   #================================================================= 
        
   #Apply h operator and transform from model space to observation space. 
   #This opearation is performed only at the end of the window.

       
   if NObsW > 0 : 
       TLoc= da_window_end #We are assuming that all observations are valid at the end of the assimilaation window.
       [YF , YFqc ] = hoperator.model_to_obs(  nx=Nx , no=NObsW , nt=1 , nens=NEns ,
                     obsloc=ObsLocW , x=stateens , obstype=ObsTypeW , obserr=ObsErrorW , obsval=YObsW ,
                     xloc=ModelConf['XLoc'] , tloc= TLoc , gross_check_factor = DAConf['GrossCheckFactor'] ,
                     low_dbz_per_thresh = DAConf['LowDbzPerThresh'] )
       YFmask = np.ones( YFqc.shape ).astype(bool)
       YFmask[ YFqc != 1 ] = False 

       ObsLocW= ObsLocW[ YFmask , : ] 
       ObsTypeW= ObsTypeW[ YFmask ] 
       YObsW= YObsW[ YFmask , : ] 
       NObsW=YObsW.size
       ObsErrorW= ObsErrorW[ YFmask , : ] 
       YF= YF[ YFmask , : ] 
             
       #=================================================================
       #  LETKF STEP  : 
       #=================================================================

       stateens = das.da_letkf( nx=Nx , nt=1 , no=NObsW , nens=NEns ,  xloc=ModelConf['XLoc']                        ,
                              tloc=da_window_end   , nvar=1                        , xfens=stateens                           ,
                              obs=YObsW        , obsloc=ObsLocW            , ofens=YF                             ,
                              rdiag=ObsErrorW  , loc_scale=DAConf['LocScalesLETKF'] , inf_coef= DAConf['InfCoefs'][0:5]  ,
                              update_smooth_coef=0.0 , temp_factor = np.ones(Nx) )[:,:,0,0]

       XA[:,:,it] = np.copy( stateens )
       PA[:,:,:,it]=PA[:,:,:,0]

#End of the DA Cycle


#=================================================================
#  VERIFICATION DIAGNOSTICS  : 
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

#=================================================================
#  SAVE OUTPUT  : 
#================================================================= 

if GeneralConf['RunSave']   :
    filename= GeneralConf['DataPath'] + '/' + GeneralConf['OutFile'] 
    print('Saving the output to ' + filename  )

    if not os.path.exists( GeneralConf['DataPath'] + '/' )  :
      os.makedirs(  GeneralConf['DataPath'] + '/'  )

    #Save Nature run output
    np.savez( filename
                     ,   XA=XA                 
                     ,   PA=PA   
                     ,   F =F
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

import matplotlib.pyplot as plt

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

   
plt.show()



