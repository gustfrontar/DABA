#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 17:50:33 2022

@author: jruiz
"""

import sys

import testconf as conf

arg_list = sys.argv

print( arg_list[:] )


print( conf.pepe1 )

exec( arg_list[1] )

print( conf.pepe1 )


