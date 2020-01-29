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
import assimilation_conf_HybridPerfectModel_R02_Den1_Freq4_Hlinear as conf         #Load the experiment configuration
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



XLoc= np.arange(40) + 0.5

stateens = np.zeros((40,20))

for i in range(40):
    for j in range(20):
        stateens[i,j]=i + j
        
           
#=================================================================
#  MAIN DATA ASSIMILATION LOOP : 
#=================================================================
start_cycle = time.time()


it = 100 


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
     

TLoc= da_window_end #We are assuming that all observations are valid at the end of the assimilaation window.

#to the observation space. 
[YF , YFmask] = hoperator.model_to_obs(  nx=Nx , no=NObsW , nt=1 , nens=NEns ,
                 obsloc=ObsLocW , x=stateens , obstype=ObsTypeW ,
                 xloc=XLoc , tloc= TLoc )
       
import matplotlib.pyplot as plt
plt.plot(YF)

