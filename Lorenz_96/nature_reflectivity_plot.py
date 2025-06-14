#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:52:08 2023
...................................

@author: jruiz
"""
import numpy as np
import matplotlib.pyplot as plt

data = np.load('./data/Nature//NatureR1_Den1_Freq4_Hradar.npz',allow_pickle=True)

Nx=data['Nx']
ObsConf=data['ObsConf']
XNature=data['XNature']
YObs=data['YObs']
ObsLoc=data['ObsLoc']
NPlot=200

#Plot the observations
tmpnobs  =int( np.arange(1,Nx+1,int(1/ObsConf['SpaceDensity']) ).size )
tmpntimes=int( np.shape(XNature)[2] )
tmpobs   =np.reshape( YObs[:,0] , [ tmpntimes , tmpnobs ]).transpose()

xobs = np.reshape( ObsLoc[:,0] , [ tmpntimes , tmpnobs ]).transpose()
tobs = np.reshape( ObsLoc[:,1] , [ tmpntimes , tmpnobs ]).transpose()



plt.figure()
plt.pcolor(tmpobs[:,-NPlot:],vmin=np.min(YObs),vmax=np.max(YObs))
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Observation location')
plt.title('Observations')
#plt.savefig( FigPath + '/' + ExpName + '/Nature_run_Y.png', facecolor='w', format='png' )
#plt.show(block=False)
#plt.close()
