# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:36:06 2017

@author: RISDA 2018
"""

#Run a tempered LETKS

import sys
sys.path.append('../model/')
sys.path.append('../data_assimilation/')

from model  import lorenzn          as model          #Import the model (fortran routines)
from obsope import common_obs       as hoperator      #Import the observation operator (fortran routines)
from da     import common_da_tools  as das            #Import the data assimilation routines (fortran routines)

import numpy as np
from scipy import stats
import os


def assimilation_letks_run( conf ) :

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
    FNature = InputData['FNature']   #Total forcing (for validation only)

    #=================================================================
    # INITIALIZATION : 
    #=================================================================
    if DAConf['AlphaTempScale'] >= 0.0 :
       print('Using AlphaTempScale to compute tempering dt')
       TempSteps = get_temp_steps( conf.DAConf['NTemp'] , conf.DAConf['AlphaTempScale'] )
    else   :
       TempSteps = DAConf['AlphaTemp']
   
    #Compute normalized pseudo_time tempering steps:
    dt_pseudo_time_vec = ( 1.0 / TempSteps ) /  np.sum( 1.0 / TempSteps ) 
    print('Tempering steps: ',TempSteps)
    print('Dt pseudo times: ',dt_pseudo_time_vec)
    
    #We set the length of the experiment according to the length of the 
    #observation array.
    
    if DAConf['ExpLength'] == None :
       DALength = int( max( ObsLoc[:,1] ) / DAConf['Freq'] )
    else:
       DALength = DAConf['ExpLength']
       XNature = XNature[:,:,0:DALength+1]
       CNature = CNature[:,:,:,0:DALength+1] 

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
    PA=np.zeros([Nx,NEns,NCoef,DALength])                   #Model parameters
    PF=np.zeros([Nx,NEns,NCoef,DALength])
    TF=np.zeros([Nx,NEns,DALength])                         #Total model forcing (small scale forcing + deterministic forcing + random forcing)
    NAssimObs=np.zeros(DALength)
    
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
    

    
    #Initialize random forcings
    CRF=np.zeros([NEns,NCoef])
    RF =np.zeros([Nx,NEns])
    
    #Initialize small scale variables and forcing
    XSS=np.zeros((NxSS,NEns))
    
    
    #Generate a random initial conditions and initialize deterministic parameters
    for ie in range(0,NEns)  :
       RandInd1=(np.round(np.random.rand(1)*DALength)).astype(int)
       RandInd2=(np.round(np.random.rand(1)*DALength)).astype(int)
    
       #Reemplazo el perturbado totalmente random por un perturbado mas inteligente.
       XA[:,ie,0]=ModelConf['Coef'][0]/2 + np.squeeze( DAConf['InitialXSigma'] * ( XNature[:,0,RandInd1] - XNature[:,0,RandInd2] ) )
         
    for ic in range(0,NCoef) :
        if DAConf['EstimateParameters'] :
            #Generate random perturbations for the initial parameter ensemble.
            PA[:,ie,ic,0]=ModelConf['Coef'][ic] + DAConf['InitialPSigma'][ic] * np.random.normal( size=Nx )
        else :    
            #Assume constant parameters
            PA[:,:,ic,:]=ModelConf['Coef'][ic] 
               
    #=================================================================
    #  MAIN DATA ASSIMILATION LOOP : 
    #=================================================================

    NormalEnd=True
    
    for it in range( 1 , DALength  )         :
       if np.mod(it,100) == 0  :
          print('Data assimilation cycle # ',str(it) )
    
    
       #=================================================================
       #  GET THE OBSERVATIONS WITHIN THE TIME WINDOW  : 
       #=================================================================
    
       #print('Observation selection')
       #start = time.time()
    
       da_window_start  = (it -1) * DAConf['Freq']
       da_window_end    = da_window_start + DAConf['Freq']
    
       #Screen the observations and get only the onew within the da window
       window_mask=np.logical_and( ObsLoc[:,1] > da_window_start , ObsLoc[:,1] <= da_window_end )
     
       ObsLocW=ObsLoc[window_mask,:]                                     #Observation location within the DA window.
       ObsTypeW=ObsType[window_mask]                                     #Observation type within the DA window
       YObsW=YObs[window_mask]                                           #Observations within the DA window
       NObsW=YObsW.size                                                  #Number of observations within the DA window
       ObsErrorW=ObsError[window_mask]                                   #Observation error within the DA window          

       #=================================================================
       #  TEMPERED LETKS  : 
       #================================================================= 
       stateens=np.zeros((Nx,NEns,1,2))
       statepens=np.zeros((Nx,NEns,NCoef,2))    
       stateens[:,:,0,0] = np.copy(XA[:,:,it-1])
       statepens[:,:,:,0] = np.copy(PA[:,:,:,it-1])
       
       for itemp in range(DAConf['NTemp'])    :
    
           #Perform initial iterations using ETKF this helps to speed up convergence.
           
           #=================================================================
           #  ENSEMBLE FORECAST  : 
           #=================================================================   
        
           ntout=int( DAConf['Freq'] / DAConf['TSFreq'] ) + 1  #Output the state every ObsFreq time steps.

           #print('Runing the ensemble')
           if ( np.any( np.isnan( stateens ) ) or np.any( np.isinf( stateens ) ) ) :
              #Stop the cycle before the fortran code hangs because of NaNs
              print('Error: The analysis contains NaN or Inf, Iteration number :',it,' will dump some variables to a temporal file')
              NormalEnd=False
              #np.savez('./tmp.npz',xf=XF[:,:,it-1],xa=XA[:,:,it-1],obs=YObsW,obsloc=ObsLocW,yf=YF)
              break
           
           [ XFtmp , XSStmp , DFtmp , RFtmp , SSFtmp , CRFtmp, CFtmp ]=model.tinteg_rk4( nens=NEns  , nt=DAConf['Freq'] ,  ntout=ntout ,
                                                   x0=stateens[:,:,0,0]  , xss0=XSS , rf0=RF    , phi=XPhi     , sigma=XSigma             ,
                                                   c0=statepens[:,:,:,0] , crf0=CRF             , cphi=CPhi    , csigma=CSigma, param=ModelConf['TwoScaleParameters'] , 
                                                   nx=Nx,  nxss=NxSS   , ncoef=NCoef  , dt=ModelConf['dt']   , dtss=ModelConf['dtss'])
                                                   
           if ( np.any( np.isnan( XFtmp[:,:,-1] ) ) or np.any( np.isinf( XFtmp[:,:,-1] ) ) ) :
              #Stop the cycle before the fortran code hangs because of NaNs
              print('Error: The forecast contains NaN or Inf, Iteration number :',it,' will dump some variables to a temporal file')
              NormalEnd=False
              break                  
                                                  
           if itemp == 0   :
              #The first iteration forecast will be stored as the forecast.
              PF[:,:,:,it] = CFtmp[:,:,:,-1]       #Store the parameter at the end of the window. 
              XF[:,:,it]=XFtmp[:,:,-1]             #Store the state variables ensemble at the end of the window.
        
           XSS=XSStmp[:,:,-1]
           CRF=CRFtmp[:,:,-1]
           RF=RFtmp[:,:,-1]
           TF[:,:,it] = SSFtmp[:,:,-1] + DFtmp[:,:,-1] + RFtmp[:,:,-1]
           
           stateens[:,:,0,:] = XFtmp[:,:,[0,-1]]    #We will store the first time and the last time.
                                                    #so the assimilation can update both.
           statepens[:,:,:,:] = CFtmp[:,:,:,[0,-1]] #We do the same for the parameters                                       

           #=================================================================
           #  OBSERVATION OPERATOR  : 
           #================================================================= 
        
           #Apply h operator and transform from model space to observation space. 
           #This opearation is performed only at the end of the window.
        
           #Set the time coordinate corresponding to the model output.                                 
           if NObsW > 0 : 
              TLoc= da_window_end #We are assuming that all observations are valid at the end of the assimilaation window.
              [YF , YFqc ] = hoperator.model_to_obs(  nx=Nx , no=NObsW , nt=1 , nens=NEns ,
                           obsloc=ObsLocW , x=stateens[:,:,:,-1] , obstype=ObsTypeW , obserr=ObsErrorW , obsval=YObsW ,
                           xloc=ModelConf['XLoc'] , tloc= TLoc , gross_check_factor = DAConf['GrossCheckFactor'] ,
                           low_dbz_per_thresh = DAConf['LowDbzPerThresh'] )
              YFmask = np.ones( YFqc.shape ).astype(bool)
              YFmask[ YFqc != 1 ] = False 

              ObsLocWStep= ObsLocW[ YFmask , : ] 
              ObsTypeWStep= ObsTypeW[ YFmask ] 
              YObsWStep= YObsW[ YFmask , : ] 
              NObsWStep=YObsWStep.size
              ObsErrorWStep= ObsErrorW[ YFmask , : ] 
              YFStep= YF[ YFmask , : ] 
                                 
           #=================================================================
           #  Compute time step in pseudo time  : 
           #=================================================================
      
           if DAConf['AddaptiveTemp']  : 
              #Addaptive time step computation
              if itemp == 0 : 
                 [a , b ] = das.da_pseudo_time_step( nx=Nx , nt=2 , no=NObsWStep , nens=NEns ,  xloc=ModelConf['XLoc']   ,
                             tloc=np.array([da_window_start,da_window_end]) , nvar=1 , obsloc=ObsLocWStep  , ofens=YFStep                         ,
                             rdiag=ObsErrorWStep , loc_scale=DAConf['LocScalesLETKF'] , niter = DAConf['NTemp']  )
              dt_pseudo_time =  a + b * (itemp + 1)
           else :
              #Equal time steps in pseudo time.  
              dt_pseudo_time = dt_pseudo_time_vec[ itemp ] * np.ones( ( Nx , 2 ) )
      
     
           #=================================================================
           #  LETKF STEP  : 
           #=================================================================
              
           temp_factor = (1.0 / dt_pseudo_time )         
           if NObsWStep > 0 :
              stateens = das.da_letkf( nx=Nx , nt=2 , no=NObsWStep , nens=NEns ,  xloc=ModelConf['XLoc']                        ,
                            tloc=np.array([da_window_start,da_window_end]) , nvar=1                        , xfens=stateens ,
                            obs=YObsWStep        , obsloc=ObsLocWStep            , ofens=YFStep                             ,
                            rdiag=ObsErrorWStep  , loc_scale=DAConf['LocScalesLETKF'] , inf_coef = DAConf['InfCoefs'][0:5]  ,
                            update_smooth_coef=0.0 , temp_factor = temp_factor )
                            
           #PARAMETER ESTIMATION
           if DAConf['EstimateParameters']   :                  
              statepens = das.da_letkf( nx=Nx , nt=2 , no=NObsWStep , nens=NEns ,  xloc=ModelConf['XLoc']      ,
                              tloc=np.array([da_window_start,da_window_end])    , nvar=NCoef                   , xfens=statepens ,
                              obs=YObsWStep          , obsloc=ObsLocWStep        , ofens=YFStep                       ,
                              rdiag=ObsErrorWStep    , loc_scale=DAConf['LocScalesP'] , inf_coef = DAConf['InfCoefsP'] ,
                              update_smooth_coef=0.0 , temp_factor = temp_factor ) 
              


       NAssimObs[it] = NObsWStep 
       #After all iterations store the final analysis.    
       XA[:,:,it] = np.copy( stateens[:,:,0,-1] )
       PA[:,:,:,it] = np.copy( statepens[:,:,:,-1] )    
    
    #=================================================================
    #  DIAGNOSTICS  : 
    #================================================================= 
    output=dict()

    SpinUp=200 #Number of assimilation cycles that will be conisdered as spin up 

    XASpread=np.std(XA,axis=1)
    XFSpread=np.std(XF,axis=1)

    XAMean=np.mean(XA,axis=1)
    XFMean=np.mean(XF,axis=1)
 
    output['YObs'] = YObs
    output['ObsType'] = ObsType
    output['ObsError'] = ObsError
    output['ObsLoc'] = ObsLoc
    output['XNature'] = XNature[:,0,0:DALength]

    output['XAMean'] = XAMean
    output['XFMean'] = XFMean

    output['XASRmse']=np.sqrt( np.mean( np.power( XAMean[:,SpinUp:DALength] - XNature[:,0,SpinUp:DALength] , 2 ) , axis=1 ) )
    output['XFSRmse']=np.sqrt( np.mean( np.power( XFMean[:,SpinUp:DALength] - XNature[:,0,SpinUp:DALength] , 2 ) , axis=1 ) )

    output['XATRmse']=np.sqrt( np.mean( np.power( XAMean - XNature[:,0,0:DALength] , 2 ) , axis=0 ) )
    output['XFTRmse']=np.sqrt( np.mean( np.power( XFMean - XNature[:,0,0:DALength] , 2 ) , axis=0 ) )

    output['XASSprd']=np.mean(XASpread,1)
    output['XFSSprd']=np.mean(XFSpread,1)

    output['XATSprd']=np.mean(XASpread,0)

    output['XASBias']=np.mean( XAMean[:,SpinUp:DALength] - XNature[:,0,SpinUp:DALength]  , axis=1 )
    output['XFSBias']=np.mean( XFMean[:,SpinUp:DALength] - XNature[:,0,SpinUp:DALength]  , axis=1 )

    output['XATBias']=np.mean(  XAMean - XNature[:,0,0:DALength]  , axis=0 )
    output['XFTBias']=np.mean(  XFMean - XNature[:,0,0:DALength]  , axis=0 )

    #OUTPUTS realted to parameter estimation    
    if DAConf['EstimateParameters'] :
       PAMean = np.mean(PA,axis=1)
       PASpread = np.std(PA,axis=1)
       TFMean   = np.mean(TF,axis=1)
       TFSpread = np.std(TF,axis=1)    

       output['PAMean']   = PAMean
       output['PASpread'] = PASpread
       output['TFMean']   = TFMean
       output['TFSpread'] = TFSpread  
       #Compute the error in the total large-scale forcing.
       output['TFSRmse']=np.sqrt( np.mean( np.power( TFMean[:,SpinUp:DALength] - FNature[:,0,SpinUp:DALength] , 2 ) , axis=1 ) )
       output['TFTRmse']=np.sqrt( np.mean( np.power( TFMean - FNature[:,0,0:DALength] , 2 ) , axis=0 ) )
       output['TFSBias']=np.mean( TFMean[:,SpinUp:DALength] - FNature[:,0,SpinUp:DALength] , axis=1 ) 
       output['TFTBias']=np.mean( TFMean - FNature[:,0,0:DALength]  , axis=0 ) 
       output['TFSSprd']=np.mean( TFSpread, axis=1 )
    
    output['Nobs'] = NAssimObs

    output['NormalEnd'] = NormalEnd

    return output


def inflation( ensemble_post , ensemble_prior , nature , inf_coefs )  :

   #This function consideres inflation approaches that are applied after the analysis. In particular when these approaches
   #are used in combination with tempering.
   DALength = nature.shape[2] - 1
   NEns = ensemble_post.shape[1]

   if inf_coefs[1] > 0.0 :
     #=================================================================
     #  RTPS  : Relaxation to prior spread (compatible with tempering iterations) 
     #=================================================================
     prior_spread = np.std( ensemble_prior , axis=1 )
     post_spread  = np.std( ensemble_post  , axis=1 )
     PostMean = np.mean( ensemble_post , axis=1 )
     EnsPert = ensemble_post - np.repeat( PostMean[:,np.newaxis] , NEns , axis=1 )
     inf_factor = ( 1.0 - inf_coefs[1] ) + ( prior_spread / post_spread ) * inf_coefs[1]
     EnsPert = EnsPert * np.repeat( inf_factor[:,np.newaxis] , NEns , axis=1 )
     ensemble_post = EnsPert + np.repeat( PostMean[:,np.newaxis] , NEns , axis=1 )

   if inf_coefs[2] > 0.0 :
     #=================================================================
     #  RTPP  : Relaxation to prior perturbations (compatible with tempering iterations) 
     #=================================================================
     PostMean = np.mean( ensemble_post , axis=1 )
     PriorMean= np.mean( ensemble_prior, axis=1 )
     PostPert = ensemble_post - np.repeat( PostMean[:,np.newaxis] , NEns , axis=1 )
     PriorPert= ensemble_prior- np.repeat( PriorMean[:,np.newaxis] , NEns , axis=1 )
     PostPert = (1.0 - inf_coefs[2] ) * PostPert + inf_coefs[2] * PriorPert 
     ensemble_post = PostPert + np.repeat( PostMean[:,np.newaxis] , NEns , axis=1 ) 

   if inf_coefs[3] > 0.0 :
     #=================================================================
     #  ADD ADDITIVE ENSEMBLE PERTURBATIONS  : 
     #=================================================================
     #Additive perturbations will be generated as scaled random
     #differences of nature run states.
     #Get random index to generate additive perturbations
     RandInd1=(np.round(np.random.rand(NEns)*DALength)).astype(int)
     RandInd2=(np.round(np.random.rand(NEns)*DALength)).astype(int)
     AddInfPert = np.squeeze( nature[:,0,RandInd1] - nature[:,0,RandInd2] ) * inf_coefs[3]
     #Shift perturbations to obtain zero-mean perturbations and add it to the ensemble.
     ensemble_post = ensemble_post + AddInfPert - np.repeat( np.mean(AddInfPert,1)[:,np.newaxis] , NEns , axis=1 )

   return ensemble_post

def get_temp_steps( NTemp , Alpha ) :
    
   #NTemp is the number of tempering steps to be performed.
   #Alpha is a slope coefficient. Larger alpha means only a small part of the information
   #will be assimilated in the first step (and the largest part will be assimilated in the last step).

   dt=1.0/float(NTemp+1)
   steps = np.exp( 1.0 * Alpha / np.arange( dt , 1.0-dt/100.0 , dt ) )
   steps = steps / np.sum(steps)

   return steps 





