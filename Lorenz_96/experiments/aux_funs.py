#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 19:57:20 2022

@author: jruiz
"""

import numpy as np

def mean_and_perts( xens , weigthed=False , w=None ) :
    
    [nvar,nens]=np.shape(xens)
    
    if weigthed  :
       mean=np.average(xens,axis=1,weights=w)
    else  :
       mean=np.mean(xens,1)
    
    perts=np.zeros( (nvar,nens) )
    for iens in range( nens ) :
        perts[:,iens] = xens[:,iens] - mean 
        
    return mean , perts
