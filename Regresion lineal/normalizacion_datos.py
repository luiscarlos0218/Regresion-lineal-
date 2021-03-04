# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:02:26 2021

@author: Luis Carlos Prada
Funcion para normalizar los datos y todos queden en unos rangos similares 
"""

import pandas as pd 
import numpy as np
def normalizacion_datos(x):
    mu=x.describe().loc['mean']
    sigma=x.describe().loc['std']
    x_norm=pd.DataFrame(np.zeros(x.shape))
    for i in range(x.shape[1]):
        for j in range(x.shape[0]):
            x_norm[i][j]=(x[i][j]-mu[i])/sigma[i]
            
    return x_norm,mu,sigma
    
    



    
    
    
    