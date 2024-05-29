#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:51:55 2024

@author: jruiz
"""
import numpy as np


def from_x_to_spec( x ) :
    
    z = np.fft.rfft( x , axis=0 ) #Apply fft for a real data
    
    phase = np.angle( z ) #The phase for each wavenumber
    
    amp = np.absolute( z ) #The amplitude for each wavenumber
    
    
    return sx 
    
def from_spec_to_x( sx ) :
    
    
    
    return x