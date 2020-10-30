# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:36:06 2017

@author: RISDA 2018
"""

#CREATE  L96-parametrized forcing spinned-up nature run. 

#Create plots of the L96 evolution.

#Add additional folders for fortran modules.

import sys
sys.path.append('../model/')
sys.path.append('../data_assimilation/')

from model   import lorenzn as model          #Import the model (fortran routines)
from obsope  import common_obs as hoperator      #Import the observation operator (fortran routines)

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import time
#import nature_conf_1scale as conf
#import nature_conf_2scale_F16 as conf
import nature_conf_2scale_F20 as conf
import os


#=================================================================
# LOAD CONFIGURATION : 
#=================================================================

GeneralConf=conf.GeneralConf
ModelConf  =conf.ModelConf
ObsConf    =conf.ObsConf
NatureConf =conf.NatureConf

#=================================================================
# INITIALIZATION : 
#=================================================================
#Get the number of parameters
NCoef=ModelConf['NCoef']
#Get the number of ensembles
NEns=NatureConf['NEns']
#Get the size of the large-scale state
Nx=ModelConf['nx']
#Get the size of the small-scale state
NxSS=ModelConf['nxss']

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

CRF0=np.zeros([NEns,NCoef])
RF0 =np.zeros([Nx,NEns])

X0=np.zeros((Nx,NEns))
XSS0=np.zeros((NxSS,NEns))
C0=np.zeros((Nx,NEns,NCoef))

#Generate a random initial conditions and initialize deterministic parameters
for ie in range(0,NEns)  :

   X0[:,ie]=ModelConf['Coef'][0]/2 + np.random.normal( size=Nx )

   for ic in range(0,NCoef) :
      C0[:,ie,ic]=ModelConf['Coef'][ic] + FSpaceAmplitude[ic]*np.cos( FSpaceFreq[ic]*2*np.pi*np.arange(0,Nx)/Nx )

#=================================================================
# RUN SPIN UP : 
#=================================================================

#Do spinup
print('Doing Spinup')
start = time.time()

 
nt=int( NatureConf['SPLength'] / ModelConf['dt'] )    #Number of time steps to be performed.
ntout=int( 2 )                       #Output only the last time.

#Runge Kuta 4 integration of the LorenzN equations
[XSU , XSSSU , DFSU , RFSU , SSFSU , CRFSU , CSU ]=model.tinteg_rk4( nens=1 , nt=nt ,  ntout=ntout                              , 
                                   x0=X0     , xss0=XSS0   , rf0=RF0     , phi=XPhi     , sigma=XSigma                          , 
                                   c0=C0     , crf0=CRF0   , cphi=CPhi   , csigma=CSigma, param=ModelConf['TwoScaleParameters'] ,
                                   nx=Nx     , ncoef=NCoef , dt=ModelConf['dt']         , dtss=ModelConf['dtss'] )
   
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


nt=int( NatureConf['Length'] / ModelConf['dt'] )                     #Number of time steps to be performed.
ntout=int( nt / ObsConf['Freq'] )  + 1                               #Output the state every ObsFreq time steps.


#Runge Kuta 4 integration of the LorenzN equations.
[XNature , XSSNature , DFNature , RFNature , SSFNature , CRFNature , CNature ]=model.tinteg_rk4( nens=1 , nt=nt ,  ntout=ntout                      ,
                                                       x0=X0     , xss0=XSS0   , rf0=RF0     , phi=XPhi     , sigma=XSigma                          ,
                                                       c0=C0     , crf0=CRF0   , cphi=CPhi   , csigma=CSigma, param=ModelConf['TwoScaleParameters'] ,
                                                       nx=Nx     , ncoef=NCoef , dt=ModelConf['dt'] , dtss=ModelConf['dtss'] )


print('Nature run took', time.time()-start, 'seconds.')

#=================================================================
# GENERATE OBSERVATIONS : 
#=================================================================

#Apply h operator and transform from model space to observation spaece.
print('Generating Observations')
start = time.time()

#Get the total number of observations
NObs = hoperator.get_obs_number( ntype=ObsConf['NetworkType'] , nx=Nx , nt=ntout ,
                                      space_density=ObsConf['SpaceDensity']  ,
                                      time_density =ObsConf['TimeDensity'] )

#Get the space time location of the observations based on the network type and 
#observation density parameters.
ObsLoc = hoperator.get_obs_location( ntype=ObsConf['NetworkType'] , nx=Nx , nt=ntout , no=NObs ,
                                      space_density=ObsConf['SpaceDensity']  , 
                                      time_density =ObsConf['TimeDensity'] )

#Assume that all observations are of the same type.
ObsType=np.ones( np.shape(ObsLoc)[0] )*ObsConf['Type']

#Set the time coordinate corresponding to the model output.
TLoc=np.arange(1,ntout+1)

#Get the observed value (without observation error)
[YObs , YObsMask]=hoperator.model_to_obs( nx=Nx   , no=NObs   , nt=ntout , nens=1    ,
                                      obsloc=ObsLoc , x=XNature , obstype=ObsType ,
                                      xloc=ModelConf['XLoc']    , tloc= TLoc )


#Get the time reference in number of time step since nature run starts.
ObsLoc[:,1]=( ObsLoc[:,1] - 1 )*ObsConf['Freq']

#Add a Gaussian random noise to simulate observation errors
ObsError=np.ones( np.shape(YObs) )*ObsConf['Error']
ObsBias =np.ones( np.shape(YObs) )*ObsConf['Bias']

YObs = hoperator.add_obs_error(no=NObs ,  nens=1  ,  obs=YObs  ,  obs_error=ObsError  ,
                               obs_bias=ObsBias , otype = ObsConf['Type'] ) 

if ObsConf['Type'] == 3 :
    YObs[ YObs < -30.0 ] = -30.0

print('Observations took', time.time()-start, 'seconds.')

#=================================================================
# SAVE THE OUTPUT : 
#=================================================================

if NatureConf['RunSave']   :
    
   FNature = DFNature + RFNature + SSFNature  #Total Nature forcing.
    
   filename=GeneralConf['DataPath'] + '/' + GeneralConf['NatureFileName']
   print('Saving the output to ' +  filename  )
   start = time.time()
    
   if not os.path.exists( GeneralConf['DataPath'] + '/' )  : 
      os.makedirs(  GeneralConf['DataPath'] + '/'  )

   #Save Nature run output
   np.savez( filename ,   XNature=XNature       , FNature=FNature
                      ,   CNature=CNature       
                      ,   YObs=YObs , NObs=NObs , ObsLoc=ObsLoc   
                      ,   ObsType=ObsType       , ObsError=ObsError 
                      ,   ModelConf=ModelConf   , NatureConf=NatureConf 
                      ,   ObsConf=ObsConf       , GeneralConf=GeneralConf 
                      ,   XSSNature=XSSNature )
   
   
   #Print XNature and XSSNature as a CSV
   #fileout=GeneralConf['DataPath'] + '/XNature.csv' 
   #np.savetxt(fileout, np.transpose( np.squeeze( XNature ) ), fmt="%6.2f", delimiter=",")
   #np.squeeze(XNature).tofile(fileout,sep=',',format='%6.2f' , newline='\n' )

   #Print XNature and XSSNature as a CSV
   #fileout=GeneralConf['DataPath'] + '/XSSNature.csv' 
   #np.savetxt(fileout, np.transpose( np.squeeze( XSSNature ) ), fmt="%6.2f", delimiter=",")

   print('Saving took ', time.time()-start, 'seconds.')

#=================================================================
# PLOT THE NATURE RUN AND THE OBSERVATIONS : 
#=================================================================

if NatureConf['RunPlot']   :

   NPlot=1000   #Plot the last NPlot times.

   print('Ploting the output')
   start = time.time()
   FigPath=GeneralConf['FigPath']
   ExpName=GeneralConf['ExpName']


   if not os.path.exists( FigPath + '/' + ExpName )  :
      os.makedirs(  FigPath + '/' + ExpName )


   #Plot the observations
   tmpnobs  =int( np.arange(1,Nx+1,int(1/ObsConf['SpaceDensity']) ).size )
   tmpntimes=int( np.shape(XNature)[2] )
   tmpobs   =np.reshape( YObs[:,0] , [ tmpntimes , tmpnobs ]).transpose()

   xobs = np.reshape( ObsLoc[:,0] , [ tmpntimes , tmpnobs ]).transpose()
   tobs = np.reshape( ObsLoc[:,1] , [ tmpntimes , tmpnobs ]).transpose()

   #Plot the nature run.
   plt.figure()
   plt.pcolor(XNature[:,0,-NPlot:],vmin=-15,vmax=15)
   plt.xlabel('Time')
   plt.ylabel('Grid points')
   plt.title('X True')
   plt.colorbar()
   plt.savefig( FigPath + '/' + ExpName + '/Nature_run_X.png', facecolor='w', format='png' )
   plt.show(block=False)
   #plt.close()


   plt.figure()
   plt.pcolor(tmpobs[:,-NPlot:],vmin=-15,vmax=15)
   plt.colorbar()
   plt.xlabel('Time')
   plt.ylabel('Observation location')
   plt.title('Observations')
   plt.savefig( FigPath + '/' + ExpName + '/Nature_run_Y.png', facecolor='w', format='png' )
   plt.show(block=False)
   #plt.close()


   plt.figure()
   plt.plot(tmpobs[0,-NPlot:],'o')
   plt.plot(XNature[0,0,-NPlot:])
   plt.xlabel('Time')
   plt.ylabel('X at 1st grid point')
   plt.title('Nature and Observations')
   plt.savefig( FigPath + '/' + ExpName + '/Nature_and_Obs_Time_Serie.png', facecolor='w', format='png' )
   plt.show(block=False)
   #plt.close()

   plt.figure()
   plt.plot(xobs[:,-1],tmpobs[:,-1],'o')
   plt.plot(np.arange(1,Nx+1),XNature[:,0,-1])
   plt.xlabel('Location')
   plt.ylabel('X at last time')
   plt.title('Nature and Observations')
   plt.savefig( FigPath + '/' + ExpName + '/Nature_and_Obs_At_Last_Time.png', facecolor='w', format='png' )
   plt.show(block=False)
   #plt.close()


   #Plot total forcing as a function of X
   plt.figure()
   tmpX=np.reshape( XNature[:,0,:] , [ XNature.shape[0] * XNature.shape[2] ] ) #A vector containing all X
   tmpF=np.reshape( FNature[:,0,:] , [ XNature.shape[0] * XNature.shape[2] ] ) #A vector containing all Forcings
   
   plt.scatter(tmpX,tmpF,s=0.5,c="g", alpha=0.5, marker='o')
   
   [a,b,r,p,std_err]= stats.linregress(tmpX,tmpF)
   
   LinearX=np.arange(round(tmpX.min())-1,round(tmpX.max())+1)
   LinearF=b + LinearX * a
   reg, =plt.plot(LinearX,LinearF,'--k')
   plt.legend([reg],['Linear Reg a=' + str(a) + ' b=' + str(b) ])
   plt.title('Forcing vs X')
   plt.savefig( FigPath + '/' + ExpName + '/Forcing_vs_X.png', facecolor='w', format='png' )
   plt.show(block=False)
   #plt.close()


   #Plot the nature run  forcing.
   plt.figure()
   plt.pcolor(FNature[:,0,-NPlot:])
   plt.xlabel('Time')
   plt.ylabel('Grid points')
   plt.title('True Forcing')
   plt.colorbar()
   plt.savefig( FigPath + '/' + ExpName + '/Nature_run_F.png', facecolor='w', format='png' )
   plt.show(block=False)
   #plt.close()


   #Plot the parameters of the deterministic forcing
   for ic in range(0,NCoef)   :
     plt.figure()
     CString=str(ic)
     plt.pcolor(np.squeeze(CNature[:,0,ic,-NPlot:]))
     plt.colorbar()
     plt.xlabel('Time')
     plt.ylabel('Grid points')
     plt.title('Parameter ' + CString )
     plt.savefig( FigPath + '/' + ExpName + '/Nature_run_Coef_' + CString + '.png', facecolor='w', format='png' )
     plt.show(block=False)
     #plt.close()
     
     plt.figure()
     plt.plot(np.mean( np.squeeze(CNature[:,0,ic,-NPlot:]) , axis = 1))
     plt.xlabel('Grid points')
     plt.ylabel('Time averaged parameter')
     plt.title('Mean parameter' + CString)
     plt.savefig( FigPath + '/' + ExpName + '/Nature_run_TimeAveCoef_' + CString + '.png', facecolor='w', format='png' )
     plt.show(block=False)
     #plt.close()

     plt.figure()
     plt.plot(np.mean(np.squeeze(CNature[:,0,ic,-NPlot:]) , axis = 0))
     plt.xlabel('Time')
     plt.ylabel('Spatially averaged parameter')
     plt.title('Mean parameter' + CString)
     plt.savefig( FigPath + '/' + ExpName + '/Nature_run_SpaceAveCoef_' + CString + '.png', facecolor='w', format='png' )
     plt.show(block=False)
     #plt.close()

   print('Ploting took ', time.time()-start, 'seconds.')

plt.show()
