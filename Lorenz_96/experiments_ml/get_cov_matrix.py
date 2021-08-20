#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Este script genera la matriz de covarianza y la matriz de correlacion 
#utilizando el metodo NMC.

"""
@author: jruiz
"""
import numpy as np
import matplotlib.pyplot as plt


ExpName = 'LETKF2scale_F20'
#Forecast_file = './data/Forecast/Forecast_LETKF2scale_F16.npz'
Forecast_file = './data/Forecast/Forecast_'+ExpName+'.npz'

Output_File = './data/Forecast/CovMat_'+ExpName+'.npz'

Data = np.load( Forecast_file ) 

NVars  = Data['XFDet'].shape[0]
NTimes = Data['XFDet'].shape[2]

#Usamos la diferencia entre dos pronosticos que verifican al mismo tiempo como proxi del error
#del pronostico.

XNatureF = Data['XNatureF']

XFError = Data['XFMean'][:,0:-1,3]-Data['XFMean'][:,1:,2]

#XFError = Data['XFMean'][:,:,2] - XNatureF[:,:,2]

CovMat = np.dot( XFError , XFError.T )/(NTimes-1)

CorMat = np.zeros( CovMat.shape )

for ii in range( CovMat.shape[0] ) :
    for jj in range( CovMat.shape[1] ) :
        
        CorMat[ii,jj] = CovMat[ii,jj]/np.sqrt(CovMat[ii,ii]*CovMat[jj,jj])
        
np.savez(Output_File,CovMat=CovMat,CorMat=CorMat)


#plt.pcolor(CovMat);plt.colorbar()
#print( np.std( XFError[:,1000:1100] , axis=1 ) ) 






