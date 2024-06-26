# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:36:06 2017

@author: RISDA 2018
"""

#Run a LETKF experiment using the observations created by the script run_nature.py

import sys
sys.path.append('../model/')
sys.path.append('../data_assimilation/')

from model  import lorenzn  as model          #Import the model (fortran routines)


import matplotlib.pyplot as plt
import numpy as np
import time
#import forecast_conf_1scale as conf             #Load the experiment configuration
import forecast_conf_2scale_F20 as conf          #Load the experiment configuration
import os

#=================================================================
# LOAD CONFIGURATION : 
#=================================================================

GeneralConf=conf.GeneralConf
ForConf    =conf.ForConf

#=================================================================
#  LOAD THE ANALYSIS AND MODEL CONFIGURATION
#=================================================================

print('Reading the analysis from file ',GeneralConf['AssimilationFile'])

InputData=np.load(GeneralConf['AssimilationFile'],allow_pickle=True)

#Copy model configuration from the assimilation experiment
ModelConf=InputData['ModelConf'][()]
DAConf   =InputData['DAConf'][()]

XA=InputData['XA']    #Initial conditions analysis
PA=InputData['PA']    #Parameter analysis

#=================================================================
#  LOAD NATURE RUN CONFIGURATION AND STATE
#=================================================================

print('Reading observations from file ',GeneralConf['NatureFile'])

InputData=np.load(GeneralConf['NatureFile'],allow_pickle=True)


#Store the true state evolution for verfication 
XNature = InputData['XNature']

#=================================================================
# INITIALIZATION : 
#=================================================================

#Number of available data assimilation cycles.
NDA=XA.shape[2]  

#Compute the number of lead times
NLeads=np.int( ForConf['ForecastLength'] / DAConf['Freq'] + 1 )

#Compute the number of forecasts that will be performed.
NForecasts=NDA - ForConf['AnalysisSpinUp'] - NLeads

#Number of analysis cycles before the first forecast.
SpinUp=ForConf['AnalysisSpinUp']

#Get the number of parameters
NCoef=ModelConf['NCoef']
#Get the size of the state vector
Nx=ModelConf['nx']
#Get the size of the small-scale state
NxSS=ModelConf['nxss']
#Get the number of ensembles (use the same size as in the assimilation cycle)
NEns=DAConf['NEns']

XF=np.zeros([Nx,NEns,NForecasts,NLeads])      #Ensemble forecasts

XFDet=np.zeros([Nx,1,NForecasts,NLeads])   #Deterministic forecasts

XNatureF=np.zeros([Nx,NForecasts,NLeads])  #Nature run reshaped so it is easiear to compare it with the forecasts.

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

C0=np.zeros((Nx,NEns,NCoef))



#=================================================================
#  MAIN FORECAST LOOP : 
#=================================================================
start = time.time()

for it in range( 0 , NForecasts  )         :

   print('Ensemble forecast for cycle # ',str(it) )

   #=================================================================
   #  ENSEMBLE FORECAST  : 
   #=================================================================   

   #Run the ensemble forecast
   #print('Runing the ensemble')

   #start = time.time()
   ntout=NLeads  #Output the state every ObsFreq time steps.
   nt=ForConf['ForecastLength']
   
   #Set the initial condition for the forecast
   XAtmp=XA[:,:,it + SpinUp ]
   #Set the parameters for the forecast
   C0=PA[:,:,:,it + SpinUp ]
   
   #If two scale model is activate then the small scale variables
   #are initialized as 0.
   XSS0=np.zeros([NxSS,NEns])

   #Run an ensemble forecast.
   [ XF[:,:,it,:] , XSS , DF , RFtmp , SSF , CRFtmp, CFtmp ]=model.tinteg_rk4( nens=NEns  , 
                                           nt=nt      , ntout=ntout   ,
                                           x0=XAtmp   , xss0=XSS0     ,
                                           rf0=RF     , 
                                           phi=XPhi   , sigma=XSigma  ,
                                           c0=C0      , crf0=CRF      ,
                                           cphi=CPhi  , csigma=CSigma ,
                                           nx=Nx      , nxss=NxSS     ,
                                           ncoef=NCoef, param=ModelConf['TwoScaleParameters'] ,
                                           dt=ModelConf['dt'] , dtss=ModelConf['dtss'] )

   #Run a deterministic forecast initialized from the ensemble mean
   XAtmp=np.zeros((Nx,1))
   XAtmp[:,0]=np.mean( XA[:,:,it + SpinUp ] , 1 )
   [ XFDet[:,0:1,it,:] , XSSDet , DFDet , RFtmpDet , SSFDet , CRFtmpDet , CFtmpDet ]= model.tinteg_rk4( nens=1  , 
                                           nt=nt      , ntout=ntout   ,
                                           x0=XAtmp   , xss0=XSS0[:,0:1]     ,
                                           rf0=RF[:,0:1], 
                                           phi=XPhi   , sigma=XSigma  ,
                                           c0=C0[:,0:1,:] , crf0=np.zeros((1,NCoef)) ,
                                           cphi=CPhi  , csigma=CSigma ,
                                           nx=Nx      , nxss=NxSS     ,
                                           ncoef=NCoef, param=ModelConf['TwoScaleParameters'] ,
                                           dt=ModelConf['dt'] , dtss=ModelConf['dtss'] )
   
   
   #Reshape XNature so it is easier to compare with the forecast.
   XNatureF[:,it,:]=XNature[:,0,it + SpinUp : it + SpinUp +NLeads]
   
print('Ensemble forecast took ', time.time()-start, 'seconds.')

#=================================================================
#  DIAGNOSTICS  : 
#================================================================= 

XFSpread=np.std(XF,axis=1)
XFMean=np.mean(XF,axis=1)

XFSRmse=np.sqrt( np.mean( np.power( XFMean - XNatureF , 2 ) , axis=1 ) )

XFTRmse=np.sqrt( np.mean( np.power( XFMean - XNatureF , 2 ) , axis=0 ) )

XFSBias=np.mean( XFMean - XNatureF , axis=1 ) 

XFTBias=np.mean( XFMean - XNatureF , axis=0 ) 

XFDetSRmse=np.sqrt( np.mean( np.power( XFDet[:,0,:,:] - XNatureF , 2 ) , axis=1 ) )

XFDetTRmse=np.sqrt( np.mean( np.power( XFDet[:,0,:,:] - XNatureF , 2 ) , axis=0 ) )

XFDetSBias=np.mean( XFDet[:,0,:,:] - XNatureF , axis=1 ) 

XFDetTBias=np.mean( XFDet[:,0,:,:] - XNatureF , axis=0 ) 

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
                     , XNatureF = XNatureF 
                     , XFMean=XFMean
                     , XFSpread=XFSpread
                     , XFSRmse=XFSRmse
                     , XFTRmse=XFTRmse 
                     , XFSBias=XFSBias
                     , XFTBias=XFTBias
                     , XFDet=XFDet
                     , XFDetSRmse=XFDetSRmse
                     , XFDetTRmse=XFDetTRmse 
                     , XFDetSBias=XFDetSBias
                     , XFDetTBias=XFDetTBias
                     , ModelConf=ModelConf   
                     , DAConf=DAConf
                     , ForConf=ForConf
                     , GeneralConf=GeneralConf )

#=================================================================
#  PLOT OUTPUT  : 
#=================================================================

#Plot analysis and forecast RMSE and spread.
if GeneralConf['RunPlot']   :

   if not os.path.exists( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] )  :
      os.makedirs(  GeneralConf['FigPath'] + '/' + GeneralConf['ExpName']  )


   #FORECAST RMSE AS A FUNCTION OF LEAD TIME

   plt.figure()
   leadrmse=np.sqrt( np.mean( np.power( XFTRmse , 2) , axis=0) )
   leadrmsedet=np.sqrt( np.mean( np.power( XFDetTRmse , 2) , axis=0) )
   leadsprd=np.sqrt( np.mean( np.mean( np.power( XFSpread , 2) , axis=1) , axis=0 ) )
   rmse, =plt.plot( leadrmse ,'b-')
   rmsedet , = plt.plot( leadrmsedet ,'b--')
   sprd, =plt.plot(  leadsprd ,'r-' )
   plt.legend([rmse,rmsedet,sprd],['RMSE ens.','RMSE det.','SPRD fcst'])
   plt.ylabel('RMSE - SPRD')
   plt.xlabel('Lead Time')
   plt.ylim( (0,leadrmsedet.max() + 0.5) )
   plt.grid(True)

   plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/RMSEvsLEAD_run.png', facecolor='w', format='png' )
   plt.show(block=False)
   #plt.close()

   plt.figure()
   leadbias=np.mean( XFTBias , axis=0)
   bias, =plt.plot(  leadbias  ,'b-')
   plt.legend([bias],['Bias fcst'])
   plt.ylabel('BIAS')
   plt.xlabel('Lead Time')
   plt.ylim( (leadbias.min()-0.5,leadbias.max()+0.5 ) )
   plt.grid(True)

   plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/BIASvsLEAD_run.png', facecolor='w', format='png' )
   plt.show(block=False)
   #plt.close()

plt.show()


##Ensemble size dependence.
#[NVar,NEns,NTimes,NLead] = XF.shape 
#
#RMSEsize = np.zeros((NEns,NLead))
#
#for iens in range(NEns)  :
#
#  #XFSpread=np.std(XF,axis=1)
#  XFMean=np.mean(XF[:,0:iens+1,:,:],axis=1)
#
#  RMSEsize[iens,:]=np.sqrt( np.mean( np.mean( np.power( XFMean - XNatureF , 2 ) , axis=1 ) , axis=0 ) )







