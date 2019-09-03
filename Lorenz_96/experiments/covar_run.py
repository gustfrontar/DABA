# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:36:06 2017

@author: RISDA 2018
"""

#Estimate and plot the covariance betweeen parameters and state variables. 
#Estimate and explore the spatio-temporal distribution of this covariance for different
#model parameters. 

import sys
sys.path.append('../model/')
sys.path.append('../data_assimilation/')

from model  import lorenznsto       as model          #Import the model (fortran routines)
from obsope import common_obs       as hoperator      #Import the observation operator (fortran routines)
from da     import common_da_tools  as das            #Import the data assimilation routines (fortran routines)

import matplotlib.pyplot as plt
import numpy as np
import time
import covar_conf_GlobalParameter as conf         #Load the experiment configuration
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

#In this case the length of the assimilation is controled in the configuration file
DALength = DAConf['DALength'] 

#DALength = 3

#Get the number of parameters
NCoef=ModelConf['NCoef']
#Get the size of the state vector
Nx=ModelConf['nx']
#Get the number of ensembles
NEns=DAConf['NEns']

#Memory allocation and variable definition.

XA=np.zeros([Nx,NEns,DALength])                         #Analisis ensemble
XF=np.zeros([Nx,NEns,DALength])                         #Forecast ensemble
PA=np.zeros([Nx,NEns,NCoef,DALength])                   #Analized parameters
PF=np.zeros([Nx,NEns,NCoef,DALength])                   #Forecasted parameters

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

C0=np.zeros((NCoef,Nx,NEns))

#Generate a random initial conditions and initialize deterministic parameters
for ie in range(0,NEns)  :

   XA[:,ie,0]=ModelConf['Coef'][0]/2 + DAConf['InitialXSigma'] * np.random.normal( size=Nx )

   for ic in range(0,NCoef) :
      if DAConf['EstimateLocalCovariance']   :
         #We will pertrub only the selected grid point.
         PA[:,ie,ic,0]=ModelConf['Coef'][ic] 
         PA[DAConf['LocalGridPoint'],ie,ic,0]=ModelConf['Coef'][ic] + DAConf['InitialPSigma'][ic] * np.random.normal( size=1 )
      else                                   :
         #We will uniformly perturb all the grid points.
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

   #start = time.time()
   ntout=int( DAConf['Freq'] / DAConf['TSFreq'] ) + 1  #Output the state every ObsFreq time steps.
   
   [ XFtmp , DF , RFtmp , CRFtmp, CFtmp ]=model.tinteg_rk4( nens=NEns  , nt=DAConf['Freq'] ,  ntout=ntout ,
                                           x0=XA[:,:,it-1]     , rf0=RF   , phi=XPhi  , sigma=XSigma,
                                           c0=PA[:,:,:,it-1]   , crf0=CRF , cphi=CPhi , csigma=CSigma,
                                           nx=Nx, ncoef=NCoef  , dt=ModelConf['dt'] )
   
   PF[:,:,:,it] = CFtmp[:,:,:,-1]       #Store the parameter at the end of the window. 
   XF[:,:,it]=XFtmp[:,:,-1]             #Store the state variables ensemble at the end of the window.

   RF=RFtmp[:,:,-1]                     #Store the random forcing state for the state variables.
   CRF=CRFtmp[:,:,-1]                   #Store the random forcing state for the parameters.

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

   XA[:,:,it] =np.squeeze( das.da_letkf( nx=Nx , nt=1 , no=NObsW , nens=NEns ,  xloc=ModelConf['XLoc']               ,
                              tloc=da_window_end    , nvar=1                        , xfens=XF[:,:,it]               ,
                              obs=YObsW             , obsloc=ObsLocW                , ofens=YF                       ,
                              rdiag=ObsErrorW       , loc_scale=DAConf['LocScales'] , inf_coefs=DAConf['InfCoefs']   ,
                              update_smooth_coef=0.0 )  )
   
   
   #PARAMETER ESTIMATION
   if DAConf['EstimateParameters']   : 
      
    if DAConf['ParameterLocalizationType'] == 1  :
       #GLOBAL PARAMETER ESTIMATION (Note that ETKF is used in this case)
   
       PA[:,:,:,it] =np.squeeze( das.da_etkf( no=NObsW , nens=NEns , nvar=3 , xfens=PF[:,:,:,it] ,
                                            obs=YObsW, ofens=YF  , rdiag=ObsErrorW   ,
                                            inf_coefs=DAConf['InfCoefsP'] ) )
       
    if DAConf['ParameterLocalizationType'] == 2  :
       #GLOBAL AVERAGED PARAMETER ESTIMATION (Parameters are estiamted locally but the agregated globally)
       #LETKF is used but a global parameter is estimated.
       
       #First estimate a local value for the parameters at each grid point.
       PA[:,:,:,it] =np.squeeze( das.da_letkf( nx=Nx , nt=1 , no=NObsW , nens=NEns ,  xloc=ModelConf['XLoc']      ,
                              tloc=da_window_end    , nvar=NCoef                    , xfens=PF[:,:,:,it]             ,
                              obs=YObsW             , obsloc=ObsLocW                , ofens=YF                       ,
                              rdiag=ObsErrorW       , loc_scale=DAConf['LocScalesP'] , inf_coefs=DAConf['InfCoefsP']   ,
                              update_smooth_coef=0.0 )  )
       
       #Spatially average the estimated parameters so we get the same parameter values
       #at each model grid point.
       for ic in range(0,NCoef)  :
           for ie in range(0,NEns)  :
              PA[:,ie,ic,it]=np.mean( PA[:,ie,ic,it] , axis = 0 )
              
   
       
    if DAConf['ParameterLocalizationType'] == 3 :
       #LOCAL PARAMETER ESTIMATION (Parameters are estimated at each model grid point and the forecast uses 
       #the locally estimated parameters)
       #LETKF is used to get the local value of the parameter.
       PA[:,:,:,it] =np.squeeze( das.da_letkf( nx=Nx , nt=1 , no=NObsW , nens=NEns ,  xloc=ModelConf['XLoc']      ,
                              tloc=da_window_end    , nvar=NCoef                    , xfens=PF[:,:,:,it]             ,
                              obs=YObsW             , obsloc=ObsLocW                , ofens=YF                       ,
                              rdiag=ObsErrorW       , loc_scale=DAConf['LocScalesP'] , inf_coefs=DAConf['InfCoefsP']   ,
                              update_smooth_coef=0.0 )  )
       
       
   else :
    #If Parameter estimation is not activated we keep the parameters as in the first analysis cycle.  
    PA[:,:,:,it]=PA[:,:,:,0]


   #print('Data assimilation took ', time.time()-start,'seconds.')

print('Data assimilation took ', time.time()-start_cycle,'seconds.')

#=================================================================
#  COVARIANCE COMPUTATION : 
#================================================================= 

CovPX=np.zeros((Nx,NCoef,DAConf['DALength']-DAConf['CovSpinUp']))

for ip in range(0,NCoef)  :
   if DAConf['InitialPSigma'][ip] > 0  :
  
      if DAConf['EstimateLocalCovariance'] :
    
         parameters=np.squeeze(PF[DAConf['LocalGridPoint'],:,ip,1])

      else   :
  
         parameters=np.squeeze(PF[0,:,ip,1])
         
      parameters=parameters-np.mean(parameters)

      for it in range(DAConf['CovSpinUp'],DAConf['DALength'])  :
          for ix in range(0,Nx)  :

              CovPX[ix,ip,it-DAConf['CovSpinUp']] = np.matmul(XF[ix,:,it]-np.mean(XF[ix,:,it]),np.transpose(parameters) )/(NEns-1)
    
CovPXMean=np.mean( CovPX , axis=2)
CovPXStd=np.std( CovPX , axis=2 )

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
                     ,   CovPX   ,  CovPXMean , CovPXStd
                     ,   ModelConf=ModelConf   , DAConf=DAConf
                     ,   GeneralConf=GeneralConf )

#=================================================================
#  PLOT OUTPUT  : 
#=================================================================

#Plot analysis and forecast RMSE and spread.
if GeneralConf['RunPlot']   :

   if not os.path.exists( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] )  :
      os.makedirs(  GeneralConf['FigPath'] + '/' + GeneralConf['ExpName']  )
      
      
   for ip in range(0,NCoef)  :
              
       if( DAConf['InitialPSigma'][ip] > 0 )   :

           CString=str(ip)
           
           #Plot mean and standard deviation of the covariance.
           
           plt.figure()
           
           mean, =plt.plot(CovPXMean[:,ip],'b-')
           std , =plt.plot(CovPXStd[:,ip],'r-')
           plt.legend([mean,std],['Cov Mean','Cov Std'])
           plt.ylabel('Mean / Std')
           plt.xlabel('Grid Point')
           plt.title('P' + CString + '-X mean and std')
           plt.ylim( (-0.1,0.1) )
           plt.grid(True)
           
           
           plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/CovPX_' + CString + '_mean_and_std.png', facecolor='w', format='png' )
           plt.show()
           
           #Plot spatiotemporal strcture of parameter-X covariance
           
           plt.figure()
           plt.pcolor(CovPX[:,ip,:])
           plt.xlabel('Time')
           plt.ylabel('Grid Point')
           plt.title('P' + CString + '-X covariance')
           plt.colorbar()
           plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/CovPX_' + CString + '.png', facecolor='w', format='png' )
           plt.show()
 
           
           
  