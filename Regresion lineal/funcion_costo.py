# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:29:59 2021

@author: Luis carlos Prada Socha 

Funcion costo del gradiente descendiente, computa el valor de la funcion costo 
"""

import numpy as np 
def funcion_costo(x,y,theta):
    
  
  # inicializamos algunos valores utiles 
      m=y.shape[0] #numero de ejemplos de entrenamiento 
    
    
      J = 0;

    #Funcion costo vectorizada 
      hipotesis=np.matmul(x,theta)
      error_cuadrado=np.power((hipotesis-y),2)
      J=(1/(2*m))*np.sum(error_cuadrado)
      
      return J
    