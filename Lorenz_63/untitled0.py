#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:52:39 2020

@author: jruiz
"""
import numpy as np
import matplotlib.pyplot as plt


#Vamos a generar un ensamble que  provienen de 2 gaussianas con diferente covarianza.


cov1=np.array([[1.0,0.7],[0.7,1.0]])
cov2=np.array([[1.0,-0.7],[-0.7,1.0]])

mean1=np.array([2,0])
mean2=np.array([-2,0])

samplesize = 20

x1 = np.random.multivariate_normal(mean1,cov1,samplesize)
x2= np.random.multivariate_normal(mean2,cov2,samplesize)


x=np.concatenate( (x1,x2), axis=0)





plt.plot(x1[:,0],x1[:,1],'o');plt.plot(x2[:,0],x2[:,1],'ro')
plt.plot(x[:,0],x[:,1],'o')