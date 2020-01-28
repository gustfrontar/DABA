#%%
#!/usr/bin/env python
# coding: utf-8
# Inicializacion. Cargamos los modulos necesarios
#Importamos todos los modulos que van a ser usados en esta notebook
from tqdm import tqdm
import numpy as np
import Lorenz_63 as model
import Lorenz_63_DA as da
import sys
from scipy.spatial.distance import cdist
sys.path.append("../Lorenz_96/data_assimilation/")
from da import common_da_tools as cdat
import matplotlib.pyplot as plt
#Seleccionar aqui el operador de las observaciones que se desea usar.
from Lorenz_63_ObsOperator import forward_operator_onlyx    as forward_operator
from Lorenz_63_ObsOperator import forward_operator_onlyx_tl as forward_operator_tl


x1=1.0
x2=38.0
x_min=1.0
x_max=40.0
dx=1

d = cdat.distance_x( x1=x1 , x2=x2 , x_min=x_min , x_max=x_max , dx=dx )

print(d)