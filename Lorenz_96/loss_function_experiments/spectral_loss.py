#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 17:59:22 2024

@author: jruiz
"""

import sys
sys.path.append('../model/')

from model  import lorenzn          as model          #Import the model (fortran routines)

import numpy as np
import os
import time
import matplotlib.pyplot as plt
import aux_functions as af

np.random.seed(20)

#=================================================================
# CONFIGURATION : 
#=================================================================

filename='./nature.npz'
#Get the number of parameters
Coef=np.array([40,0,0])                    #Coefficient of parametrized forcing (polynom coefficients starting from coef[0]*x^0 + coef[1]*x ... )
NCoef=np.size(Coef)           #Get the total number of coefs.
TwoScaleParameters=np.array([10,10,1])     #Small scale and coupling parameters C , B and Hint
                                                        #Set Hint /= 0 to enable two scale model
forecast_lead=20  #Time steps to forecast.                                                         
dt=0.005 
dtss=dt/10.0 
outputfreq=4  #output frequency in number of time steps.   
SPLength=100  #Spin up length
Length=400   #Simulation length                                        
#Get the number of ensembles
NEns=1
#Get the size of the large-scale state
nx=20
nxss=nx*8
#Get the size of the small-scale state
NxSS=80

#Initialize model configuration, parameters and state variables.
XSigma=0.0
XPhi=1.0
CSigma=np.zeros(NCoef)
CPhi=1.0

#=================================================================
# INITIALIZATION : 
#=================================================================

FSpaceAmplitude=np.zeros(NCoef) 
CRF0=np.zeros([NEns,NCoef])
RF0 =np.zeros([nx,NEns])
X0=np.zeros((nx,NEns))
XSS0=np.zeros((nxss,NEns))
C0=np.zeros((nx,NEns,NCoef))

#Generate a random initial conditions and initialize deterministic parameters
for ie in range(0,NEns)  :

   X0[:,ie]=Coef[0]/2 + np.random.normal( size=nx )

   for ic in range(0,NCoef) :
      C0[:,ie,ic]= Coef[ic] 
#=================================================================
# RUN SPIN UP : 
#=================================================================

#Do spinup
print('Doing Spinup')
start = time.time()

 
nt=int( SPLength / dt )    #Number of time steps to be performed.
ntout=int( 2 )                       #Output only the last time.

#Runge Kuta 4 integration of the LorenzN equations
[XSU , XSSSU , DFSU , RFSU , SSFSU , CRFSU , CSU ]=model.tinteg_rk4( nens=1 , nt=nt ,  ntout=ntout                              , 
                                   x0=X0     , xss0=XSS0   , rf0=RF0     , phi=XPhi     , sigma=XSigma                          , 
                                   c0=C0     , crf0=CRF0   , cphi=CPhi   , csigma=CSigma, param=TwoScaleParameters              ,
                                   nx=nx     , ncoef=NCoef , dt=dt         , dtss=dtss )
   
print('Spinup up took', time.time()-start, 'seconds.')


#=================================================================
# RUN NATURE : 
#=================================================================

#Run nature
print('Doing Nature Run')
start = time.time()

X0=XSU[:,:,-1]                  #Start large scale variables from the last time of the spin up run.
XSS0=XSSSU[:,:,-1]              #Start small scale variables from the last time of the sipin up run.  
CRF0=CRFSU[:,:,-1]              #Spin up for the random forcing for the parameters   
RF0=RFSU[:,:,-1]                #Spin up for the random forcing for the state variables


nt=int( Length / dt )                     #Number of time steps to be performed.
ntout= int( nt / outputfreq ) + 1 


#Runge Kuta 4 integration of the LorenzN equations.
[XNature , XSSNature , DFNature , RFNature , SSFNature , CRFNature , CNature ]=model.tinteg_rk4( nens=1 , nt=nt ,  ntout=ntout           ,
                                                        x0=X0     , xss0=XSS0   , rf0=RF0     , phi=XPhi      , sigma=XSigma              ,
                                                        c0=C0     , crf0=CRF0   , cphi=CPhi   , csigma=CSigma , param=TwoScaleParameters  ,
                                                        nx=nx     , ncoef=NCoef , dt=dt       , dtss=dtss )


print('Nature run took', time.time()-start, 'seconds.')

#=================================================================
# SAVE AND PLOT NATURE RUN: 
#=================================================================

print('Saving the output to ' +  filename  )
start = time.time()
    
#Save Nature run output
#np.savez( filename , XNature=XNature , XSSNature=XSSNature )
   
print('Saving took ', time.time()-start, 'seconds.')

#Plot the nature run.
plt.figure()
plt.pcolor(XNature[:,0,-200:],vmin=-50,vmax=50)
plt.xlabel('Time')
plt.ylabel('Grid points')
plt.title('X True')
plt.colorbar()
plt.show(block=False)    

plt.figure()
freq , bins =np.histogram( XNature.flatten() , density=True )
plt.plot( 0.5 * (bins[1:] + bins[:-1]) , np.log(freq) )
plt.xlabel('X')
plt.title('Log frequency')
plt.show(block=False)    

#=================================================================
# GENERATE DATA SET FOR  ML 
#=================================================================

XInput  = XNature[:,0,0:-forecast_lead]
XTarget = XNature[:,0,forecast_lead:]

#=================================================================
# TRANSFORM DATA SET  
#=================================================================

ZTarget = np.fft.rfft( XTarget , axis=0 ) #Apply fft for a real data
PhaseTarget = np.angle( ZTarget ) #The phase for each wavenumber
AmpTarget   = np.absolute( ZTarget ) #The amplitude for each wavenumber
    







    
    