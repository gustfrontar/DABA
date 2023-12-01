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

import numpy as np
from scipy import stats
import os


def assimilation_hybrid_run( conf ) :

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
    if DAConf['AlphaTempScale'] > 0.0 :
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
       da_analysis_time = da_window_end
    
       #Screen the observations and get only the onew within the da window
       window_mask=np.logical_and( ObsLoc[:,1] > da_window_start , ObsLoc[:,1] <= da_window_end )
     
       ObsLocW=ObsLoc[window_mask,:]                                     #Observation location within the DA window.
       ObsTypeW=ObsType[window_mask]                                     #Observation type within the DA window
       YObsW=YObs[window_mask]                                           #Observations within the DA window
       NObsW=YObsW.size                                                  #Number of observations within the DA window
       ObsErrorW=ObsError[window_mask]                                   #Observation error within the DA window          

       #=================================================================
       #  TEMPERED RUNNING IN PLACE  : 
       #================================================================= 
       stateens=np.zeros((Nx,NEns,1,2))    
       stateens[:,:,0,0] = np.copy(XA[:,:,it-1])
       statepfens = np.copy(PA[:,:,:,it-1])
       
       #statefens = np.zeros((Nx,NEns,1,2))
       
       for irip in range(DAConf['NRip'])    :
    
           #Perform initial iterations using ETKF this helps to speed up convergence.
           
           #=================================================================
           #  ENSEMBLE FORECAST  : 
           #=================================================================   
        
           ntout=int( DAConf['Freq'] / DAConf['TSFreq'] ) + 1  #Output the state every ObsFreq time steps.

           #print('Runing the ensemble')
           if np.any( np.isnan( stateens ) ) :
               #Stop the cycle before the fortran code hangs because of NaNs
               print('Error: The analysis contains NaN, Iteration number :',it)
               break
           
           [ XFtmp , XSStmp , DFtmp , RFtmp , SSFtmp , CRFtmp, CFtmp ]=model.tinteg_rk4( nens=NEns  , nt=DAConf['Freq'] ,  ntout=ntout ,
                                                   x0=stateens[:,:,0,0]  , xss0=XSS , rf0=RF    , phi=XPhi     , sigma=XSigma             ,
                                                   c0=statepfens   , crf0=CRF             , cphi=CPhi    , csigma=CSigma, param=ModelConf['TwoScaleParameters'] , 
                                                   nx=Nx,  nxss=NxSS   , ncoef=NCoef  , dt=ModelConf['dt']   , dtss=ModelConf['dtss'])
           if irip == 0   :
              #The first iteration forecast will be stored as the forecast.
              PF[:,:,:,it] = CFtmp[:,:,:,-1]       #Store the parameter at the end of the window. 
              XF[:,:,it]=XFtmp[:,:,-1]             #Store the state variables ensemble at the end of the window.
        
        
           F[:,:,it] =DFtmp[:,:,-1]+RFtmp[:,:,-1]+SSFtmp[:,:,-1]  #Store the total forcing 
           
           XSS=XSStmp[:,:,-1]
           CRF=CRFtmp[:,:,-1]
           RF=RFtmp[:,:,-1]
           
           stateens[:,:,0,:] = XFtmp[:,:,[0,-1]]  #We will store the first time and the last time.
                                          #so the assimilation can update both.
           statepfens = CFtmp[:,:,:,-1]   #Parameters do not change in time within the assimilation window
                                          #so we only need to store their value at the end of the window.
        
        
        
           if it < DAConf['NKalmanSpinUp']  :
               BridgeParam = 0.0  #Force pure Kalman step.
           else                             :
               BridgeParam = DAConf['BridgeParam']

           #print('Runing the ensemble')
           if np.any( np.isnan( stateens ) ) :
              #Stop the cycle before the fortran code hangs because of NaNs
              print('Error: The analysis contains NaN, Iteration number :',it)
              break

                      
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
                                 obsloc=ObsLocW , x=stateens[:,:,:,-1] , obstype=ObsTypeW ,
                                 xloc=ModelConf['XLoc'] , tloc= TLoc )
               
           #=================================================================
           #  Compute time step in pseudo time  : 
           #=================================================================
      
           if DAConf['EnableTempering']  :
              #Enable tempering 
              if DAConf['AddaptiveTemp']  : 
                 #Addaptive time step computation
                 if irip == 0 : 
                    #local_obs_error = ObsErrorW * DAConf['NTemp'] / ( 1.0 - BridgeParam ) 
                    [a , b ] = das.da_pseudo_time_step( nx=Nx , nt=2 , no=NObsW , nens=NEns ,  xloc=ModelConf['XLoc']   ,
                            tloc=np.array([da_window_start,da_window_end]) , nvar=1 , obsloc=ObsLocW  , ofens=YF    ,
                            rdiag=ObsErrorW , loc_scale=DAConf['LocScalesLETKF'] , niter = DAConf['NRip']  )
                 dt_pseudo_time =  a + b * (irip + 1)
                 
              else :
                  #Equal time steps in pseudo time.  
                  dt_pseudo_time = np.ones((Nx,2)) / DAConf['NRip']   

           else                   :
             #Original RIP formulation no tempering is enable.  
             dt_pseudo_time = np.ones((Nx,2)) 

           #=================================================================
           #  LETKF STEP  : 
           #=================================================================
              
           if BridgeParam < 1.0 :
              #Compute the tempering parameter.
              temp_factor = ( 1.0 / dt_pseudo_time ) / ( 1.0 - BridgeParam )  
              
              stateens = das.da_letkf( nx=Nx , nt=2 , no=NObsW , nens=NEns ,  xloc=ModelConf['XLoc']                ,
                          tloc=np.array([da_window_start,da_window_end])    , nvar=1 , xfens=stateens               ,
                          obs=YObsW             , obsloc=ObsLocW                , ofens=YF                          ,
                          rdiag=ObsErrorW , loc_scale=DAConf['LocScalesLETKF'] , inf_coefs=DAConf['InfCoefs']       ,
                          update_smooth_coef=0.0 , temp_factor = temp_factor )
        
           #=================================================================
           #  OBS OPERATOR AND ETPF STEP  : 
           #=================================================================
                 
           if BridgeParam > 0.0 :
              prior_weights = np.ones((Nx,NEns,2))/NEns  #Resampling is performed at each time step. So assume equal weigths 
                                                      #for the prior.
              #Compute the tempering parameter.
              TLoc= da_window_end #We are assuming that all observations are valid at the end of the assimilaation window.
              [YF , YFmask] = hoperator.model_to_obs(  nx=Nx , no=NObsW , nt=1 , nens=NEns ,
                                 obsloc=ObsLocW , x=stateens[:,:,:,-1] , obstype=ObsTypeW ,
                                 xloc=ModelConf['XLoc'] , tloc= TLoc )

              temp_factor = (1.0 / dt_pseudo_time ) / ( BridgeParam )         
              [stateens , wa]= das.da_letpf( nx=Nx , nt=2 , no=NObsW , nens=NEns ,  xloc=ModelConf['XLoc']         , 
                          tloc=np.array([da_window_start,da_window_end]) , nvar=1  , xfens=stateens                , 
                          obs=YObsW             , obsloc=ObsLocW                , ofens=YF                         ,
                          rdiag=ObsErrorW , loc_scale=DAConf['LocScalesLETPF'] ,  temp_factor = temp_factor        ,
                          multinf=DAConf['InfCoefs'][0] , w_in = prior_weights )
              
              
                                          
           #PARAMETER ESTIMATION
           #FOR THE MOMENT PURE LETKF IS BEING USED TO UPDATE PARAMETERS
           if DAConf['EstimateParameters']   : 
              
            if DAConf['ParameterLocalizationType'] == 1  :
               #GLOBAL PARAMETER ESTIMATION (Note that ETKF is used in this case)
               local_obs_error = ObsErrorW * DAConf['NRip'] 
               statepfens = das.da_etkf( no=NObsW , nens=NEns , nvar=NCoef , xfens=statepfens ,
                                                    obs=YObsW, ofens=YF  , rdiag=ObsErrorW   ,
                                                    inf_coefs=DAConf['InfCoefsP'] )[:,:,:,0] 
 
            if DAConf['ParameterLocalizationType'] == 2  :
               #GLOBAL AVERAGED PARAMETER ESTIMATION (Parameters are estiamted locally but the agregated globally)
               #LETKF is used but a global parameter is estimated.
               
               #First estimate a local value for the parameters at each grid point.
               statepfens = das.da_letkf( nx=Nx , nt=1 , no=NObsW , nens=NEns ,  xloc=ModelConf['XLoc']                        ,
                                      tloc=da_window_end    , nvar=NCoef                    , xfens=statepfens                 ,
                                      obs=YObsW             , obsloc=ObsLocW                , ofens=YF                         ,
                                      rdiag=ObsErrorW       , loc_scale=DAConf['LocScalesP'] , inf_coefs=DAConf['InfCoefsP']   ,
                                      update_smooth_coef=0.0 )[:,:,:,0]
               
               #Spatially average the estimated parameters so we get the same parameter values
               #at each model grid point.
               for ic in range(0,NCoef)  :
                   for ie in range(0,NEns)  :
                      statepfens[:,ie,ic,it]=np.mean( statepfens[:,ie,ic,it] , axis = 0 )
                      
            if DAConf['ParameterLocalizationType'] == 3 :
               #LOCAL PARAMETER ESTIMATION (Parameters are estimated at each model grid point and the forecast uses 
               #the locally estimated parameters)
               #LETKF is used to get the local value of the parameter.
               statepfens = das.da_letkf( nx=Nx , nt=1 , no=NObsW , nens=NEns ,  xloc=ModelConf['XLoc']      ,
                                      tloc=da_window_end    , nvar=NCoef                    , xfens=statepfens                 ,
                                      obs=YObsW             , obsloc=ObsLocW                , ofens=YF                         ,
                                      rdiag=ObsErrorW       , loc_scale=DAConf['LocScalesP'] , inf_coefs=DAConf['InfCoefsP']   ,
                                      update_smooth_coef=0.0 )[:,:,:,0]
               
               
           else :
            #If Parameter estimation is not activated we keep the parameters as in the first analysis cycle.  
            statepfens=statepfens
    
     
       #After all RIP iterations store the final analysis.    
       XA[:,:,it] = np.copy( stateens[:,:,0,-1] )    
       PA[:,:,:,it] = np.copy( statepfens )
    
    #=================================================================
    #  DIAGNOSTICS  : 
    #================================================================= 
    output=dict()
    
    SpinUp=200 #Number of assimilation cycles that will be conisdered as spin up 
    
    XASpread=np.std(XA,axis=1)
    XFSpread=np.std(XF,axis=1)
    
    XAMean=np.mean(XA,axis=1)
    XFMean=np.mean(XF,axis=1)
    
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
    
    output['Nobs'] = NAssimObs

    return output





