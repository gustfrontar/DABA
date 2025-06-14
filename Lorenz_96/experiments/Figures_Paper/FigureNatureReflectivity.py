#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:52:08 2023

@author: jruiz
"""
import sys
sys.path.append('/home/jruiz/code/python/common_python/common_modules/') 

import numpy as np
import matplotlib.pyplot as plt
import common_plot_functions as cpf

data = np.load('../data/Nature/NatureR1_Den1_Freq4_Hradar.npz',allow_pickle=True)

ObsConf=data['ObsConf']
XNature=data['XNature']
YObs=data['YObs']
ObsLoc=data['ObsLoc']
NPlot=200

#Plot the observations
tmpnobs  = XNature.shape[0]
tmpntimes= XNature.shape[2]
tmpobs   =np.reshape( YObs[:,0] , [ tmpntimes , tmpnobs ]).transpose()

xobs = np.reshape( ObsLoc[:,0] , [ tmpntimes , tmpnobs ]).transpose()
tobs = np.reshape( ObsLoc[:,1] , [ tmpntimes , tmpnobs ]).transpose()



fig , axs = plt.subplots( 1 , 2 , figsize=(12,5),sharey=True)

my_map2 = cpf.cmap_discretize('gist_ncar',100)
my_map1 = cpf.cmap_discretize('bwr',100)
clevs1 = np.arange( -16 , 16 , 0.5 )
clevs2 = np.arange( 0.0 , 60.0 , 1 )
cmap1=axs[0].contourf(XNature[:,0,-NPlot:],clevs1,cmap=my_map1)
axs[0].set_ylabel('Observation location')
axs[0].set_xlabel('Time')
axs[0].set_title('(a)')
cbar1=fig.colorbar(cmap1,ax=axs[0])
cbar1.set_label('X')
cmap2=axs[1].contourf(tmpobs[:,-NPlot:],clevs2,cmap=my_map2)
cbar2=fig.colorbar(cmap2,ax=axs[1])
cbar2.set_label('Reflectivity')
axs[1].set_xlabel('Time')
axs[1].set_title('(b)')
plt.xlabel('Time')
plt.savefig( './FigureNatureReflectivity.png', facecolor='w', format='png' )
plt.show(block=False)
plt.close()

