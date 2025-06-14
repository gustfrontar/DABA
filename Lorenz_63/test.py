#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: juan
"""


import numpy as np

import matplotlib.pyplot as plt

ntimes=100

omega_sq=2.0

dt=0.1

estado=np.zeros((ntimes,2))

estado[0,:]=np.array([0.0,1.0])

M=np.array([[1.0-(dt*dt)*omega_sq , dt ],[-dt*omega_sq , 1.0]])

for it in range(1,ntimes)  :
    
    estado[it,:] = np.dot(M,estado[it-1,:])
    
    
plt.plot( estado[:,0] )



    




    