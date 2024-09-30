# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:36:06 2017

@author: RISDA 2018
"""

#Estimate model sensitivity to selected parameters

import sys
sys.path.append('../model/')
sys.path.append('../data_assimilation/')

from model  import lorenzn       as model          #Import the model (fortran routines)

import matplotlib.pyplot as plt
import numpy as np
import time
import sensitivity_conf_PerfectInitialConditions as conf      #Load the experiment configuration
import os

#=================================================================
# LOAD CONFIGURATION : 
#=================================================================

GeneralConf         =conf.GeneralConf
SensitivityConf     =conf.SensitivityConf
ModelConf           =conf.ModelConf

#=================================================================
#  LOAD THE ANALYSIS AND MODEL CONFIGURATION
#=================================================================

print('Reading the analysis from file ',GeneralConf['AssimilationFile'])

InputData=np.load(GeneralConf['AssimilationFile'],allow_pickle=True)

#Copy model configuration from the assimilation experiment
ModelConf=InputData['ModelConf'][()]
DAConf   =InputData['DAConf'][()]

XAMean=InputData['XAMean']    #Analysis ensemble mean
PA=InputData['PA']            #Parameter analysis

#=================================================================
#  LOAD NATURE RUN CONFIGURATION AND STATE
#=================================================================

print('Reading the nature run from file ',GeneralConf['NatureFile'])

InputData=np.load(GeneralConf['NatureFile'],allow_pickle=True)

#Store the true state evolution for verfication 
XNature = InputData['XNature']

#=================================================================
# INITIALIZATION : 
#=================================================================

#Get the number of parameters
NCoef=ModelConf['NCoef']
#Get the size of the state vector
Nx=ModelConf['nx']
#Get the size of the small scale state vector
NxSS=ModelConf['nxss']
#Get the number of ensembles. Note that to explore the sensitivity all ensemble
#members will share the same initial conditions but they will use a different parameter
#value. The value associated to each ensemble member will be the same troughout 
#all the sensitivity estimation experiment.
NEns=SensitivityConf['NP']  #One ensemble member for each parameter value.

#Number of data assimilation cycles before starting the computation of the sensitivity
SpinUp=SensitivityConf['AnalysisSpinUp']

#Number of available data assimilation cycles.
NDA=XAMean.shape[1]  

#Compute the number of lead times
NLeads=np.int( SensitivityConf['ForecastLength'] / DAConf['Freq'] + 1 )

#Compute the number of forecasts that will be performed.
NForecasts=NDA - SensitivityConf['AnalysisSpinUp'] - NLeads

XF=np.zeros([Nx,NEns,NForecasts,NLeads])   #Ensemble forecasts


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

#Initialize parameters
for ic in range(0,NCoef)  :
    C0[:,:,ic]=ModelConf['Coef'][ic]


#Perturb the selected parameter, assign a different parameter value to each 
#ensemble member in increasing order from PMin to PMax.
for ip in range(0,SensitivityConf['NP'])  :
    
   C0[:,ip,SensitivityConf['PIndex']]=SensitivityConf['PVals'][ip]
   
XAtmp = np.zeros([Nx,NEns])
 
#=================================================================
#  MAIN FORECAST LOOP : 
#=================================================================

for it in range( 0 , NForecasts  )         :

   print('Forecast cycle # ',str(it) )

   #=================================================================
   #  ENSEMBLE FORECAST  : 
   #=================================================================   

   #Run the ensemble forecast
   #print('Runing the ensemble')

   #start = time.time()
   ntout=NLeads  #Output the state every ObsFreq time steps.
   nt=SensitivityConf['ForecastLength']
   
   #Set the initial conditions. All ensemble members will share the same 
   #initial conditions. These can be perfect (i.e. taken from the nature run) or 
   #imperfect (i.e. taken from the analysis)
   if SensitivityConf['UseNatureRunAsIC']    :
       #Using perfect initial conditions
       for ie in range(0,NEns)  :
         XAtmp[:,ie]=XNature[:,0,it+SpinUp]
   else                                      :
       #Using imperfect initial conditions
       for ie in range(0,NEns)  :
         XAtmp[:,ie]=XAMean[:,it + SpinUp ]
   
   #If two scale model is activate then the small scale variables
   #are initialized as 0.
   XSS0=np.zeros([NxSS,NEns])

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

   #We reshape XNature so it will be easier to compute error metrics with respect
   #to the forecast performed with different parameter value.
   XNatureF[:,it,:]=XNature[:,0,it + SpinUp : it + SpinUp +NLeads]
   
   #print('Ensemble forecast took ', time.time()-start, 'seconds.')

#=================================================================
#  DIAGNOSTICS  : 
#================================================================= 

XFSpread=np.std(XF,axis=1)
XFMean=np.mean(XF,axis=1)

#Compute the RMSE as a function of the lead time and the parameter value (ie
#compute RMSE as for each ensemble member.)

XFSRmse=np.zeros((Nx,NEns,NLeads))
XFTRmse=np.zeros((NEns,NForecasts,NLeads))

XFSBias=np.zeros((Nx,NEns,NLeads))
XFTBias=np.zeros((NEns,NForecasts,NLeads))

for ie in range( 0 , NEns  )         :
    
    XFSRmse[:,ie,:]=np.sqrt( np.mean( np.power( XF[:,ie,:,:] - XNatureF , 2 ) , axis=1 ) )
    
    XFTRmse[ie,:,:]=np.sqrt( np.mean( np.power( XF[:,ie,:,:] - XNatureF , 2 ) , axis=0 ) )

    XFSBias[:,ie,:]=np.mean( XF[:,ie,:,:] - XNatureF , axis=1 ) 

    XFTBias[ie,:,:]=np.mean( XF[:,ie,:,:] - XNatureF , axis=0 ) 


XFTotMse= np.mean( np.mean( np.power( XFTRmse[:,:,1:] , 2) , axis=2) , axis=1 ) 

XFTotBias= np.mean( np.mean( XFTBias[:,:,1:]  , axis=2) , axis=1 )

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
                     , XFTotMse=XFTotMse
                     , XFSRmse=XFSRmse
                     , XFTRmse=XFTRmse 
                     , XFSBias=XFSBias
                     , XFTBias=XFTBias
                     , XFTotBias=XFTotBias
                     , ModelConf=ModelConf   
                     , DAConf=DAConf
                     , SensitivityConf=SensitivityConf
                     , GeneralConf=GeneralConf )

#=================================================================
#  PLOT OUTPUT  : 
#=================================================================


#GRAFICAR LAS FUNCIONES DE COSTO.
#Plot analysis and forecast RMSE and spread.
if GeneralConf['RunPlot']   :

   if not os.path.exists( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] )  :
      os.makedirs(  GeneralConf['FigPath'] + '/' + GeneralConf['ExpName']  )


   #FORECAST RMSE AS A FUNCTION OF LEAD TIME

   plt.figure()

   plt.plot( SensitivityConf['PVals'] , XFTotMse ,'b-')
   plt.ylabel('MSE')
   plt.xlabel('Parameter Value')
   plt.ylim( ( 0, XFTotMse.max() ) )
   plt.grid(True)

   plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/SensitivityMse.png', facecolor='w', format='png' )
   plt.show(block=False)
   #plt.close()
  
   
   plt.figure()

   plt.plot( SensitivityConf['PVals'] , XFTotBias ,'b-')
   plt.ylabel('Bias')
   plt.xlabel('Parameter Value')
   plt.ylim( (XFTotBias.min(),XFTotBias.max() ) )
   plt.grid(True)

   plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/SensitivityBias.png', facecolor='w', format='png' )
   plt.show(block=False)
   #plt.close()

   plt.figure()
   
   plt.contourf( np.arange(1,NLeads) , SensitivityConf['PVals'] , np.log( np.mean( np.power( XFTRmse[:,:,1:] , 2) , axis=1 ) ) )
   plt.ylabel('Parameter value')
   plt.xlabel('Forecast Lead Time')
   plt.title('log(MSE) as a function of parameter and lead time')

   plt.grid(True)

   plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/SensitivityMseLeadTime.png', facecolor='w', format='png' )   
   plt.show(block=False)
   #plt.close()

   plt.figure()
   
   plt.contourf( np.arange(1,NLeads) , SensitivityConf['PVals'] , np.mean( XFTBias[:,:,1:] , axis=1 ) )
   plt.ylabel('Parameter value')
   plt.xlabel('Forecast Lead Time')
   plt.title('Bias as a function of parameter and lead time')

   plt.grid(True)

   plt.savefig( GeneralConf['FigPath'] + '/' + GeneralConf['ExpName'] + '/SensitivityBiasLeadTime.png', facecolor='w', format='png' )   
   plt.show(block=False)
   #plt.close()

plt.show()
