#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jruiz
"""
import numpy as np
import matplotlib.pyplot as plt


#Forecast_file = './data/Forecast/Forecast_LETKF2scale_F16.npz'
Forecast_file = './data/Forecast/Forecast_LETKF1scale.npz'
Lead_time = 100 

Data = np.load( Forecast_file ) 

NVars  = Data['XFDet'].shape[0]
NTimes = Data['XFDet'].shape[2]

XFDet = Data['XFDet'][:,0,:,Lead_time].reshape( NVars * NTimes )
XFMean = Data['XFMean'][:,:,Lead_time].reshape( NVars * NTimes )

XNatureF = Data['XNatureF'][:,:,Lead_time].reshape( NVars * NTimes )


EDet = XFDet - XNatureF
EMean= XFMean - XNatureF

XFBin = np.arange( -6 , 12 , 1.0 )
EDetBin = np.zeros( XFBin.size - 1)
EMeanBin= np.zeros( XFBin.size - 1)

for ii in range( XFBin.size - 1 ) :
    BinMask = np.logical_and( XFDet > XFBin[ii] , XFDet <= XFBin[ii+1] )
    EDetBin[ii] = np.mean( EDet[BinMask] )
    BinMask = np.logical_and( XFMean > XFBin[ii] , XFMean <= XFBin[ii+1] )
    EMeanBin[ii] = np.mean( EMean[BinMask] )
XFBinPlot = 0.5 * ( XFBin[0:-1] + XFBin[1:] )



plt.figure()
plt.plot( XFDet , EDet , 'o' )
plt.plot( XFBinPlot , EDetBin , '-r')
plt.grid()
plt.xlim([-6,12])
plt.ylim([-15,15])


plt.figure()
plt.plot( XFMean , EMean ,'o')
plt.plot( XFBinPlot , EMeanBin , '-r')
plt.grid()
plt.xlim([-6,12])
plt.ylim([-15,15])



