#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:50:17 2020
@author: jruiz
"""
import pickle
import sys
sys.path.append('../model/')
sys.path.append('../data_assimilation/')

import numpy as np
import nature_module as nature
import default_nature_conf as conf

####################################
# PARAMETERS THAT WILL BE ITERATED
####################################

conf.ModelConf['Coef'] = np.array([16])
conf.ModelConf['NCoef'] = np.size( conf.ModelConf['Coef'] )
conf.ModelConf['TwoScaleParameters'] = np.array([10,10,1])
conf.ModelConf['nxss'] = conf.ModelConf['nx'] * 32
conf.ModelConf['dtss'] = conf.ModelConf['dt'] / 10

FreqList=[4,12,20]
SpaceDensityList=[0.5,1.0]
ObsOpe=[1,3]
ObsError=[1,5] 
conf.GeneralConf['RandomSeed'] = 10  #Fix random seed


for MyFreq in FreqList :
    for MyDen in SpaceDensityList :
       for MyOO in ObsOpe :
          for MyOE in ObsError :

              conf.ObsConf['Freq'] = MyFreq  
              conf.ObsConf['SpaceDensity'] = MyDen
              conf.ObsConf['Type'] = MyOO
              conf.ObsConf['Error'] = MyOE
              conf.GeneralConf['ExpName']='Nature_Freq' + str(MyFreq) + '_Den' + str(MyDen) + '_Type' + str(MyOO) + '_ObsErr' + str(MyOE)  
              conf.GeneralConf['NatureFileName']='MultipleNature_' + conf.GeneralConf['ExpName'] + '_twoscales.npz'
              nature.nature_run( conf )
            
    


