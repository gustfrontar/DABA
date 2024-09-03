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

FreqList=[4,8,12,16,20]
SpaceDensityList=[0.1,0.25,0.5,1.0]
ObsOpe=[1,3]
ObsError=[1,5,25] 
conf.GeneralConf['RandomSeed'] = 10  #Fix random seed

for MyFreq in FreqList :
    for MyDen in SpaceDensityList :
       for MyOO in ObsOpe :
          for MyOE in ObsError :

              conf.ObsConf['Freq'] = MyFreq  
              conf.ObsConf['SpaceDensity'] = MyDen
              conf.ObsConf['Type'] = MyOO
              conf.ObsConf['ObsError'] = MyOE
              conf.GeneralConf['NatureFileName']='Nature_Freq' + str(MyFreq) + '_Den' + str(MyDen) + '_Type' + str(MyOO) + '_ObsErr' + str(MyOE) + '.npz' 

              nature.nature_run( conf )
            
    


