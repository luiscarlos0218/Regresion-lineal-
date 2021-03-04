# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 18:27:50 2021

@author: Luis Carlos Prada 
Funcion para ejecutar el gradiente descendiente del algoritmo regresion para obtener
los parametros de aprendizaje theta
"""
import numpy as np
from funcion_costo import funcion_costo
def gradiente_descendiente(x,y,theta,alpha,num_iteraciones):
    
    #Variables iniciales necesarias
    m=y.shape[0]
    J_historia=np.zeros([num_iteraciones,1])
    num_parametros=len(theta)
    
    for iteracion in range(num_iteraciones):
        
        theta_temporal=np.zeros([num_parametros]) #inicializo en 0  al vector theta en cada una de las iteraciones 
        for l in range(num_parametros):
            theta_temporal[l]=theta[l] #Primero asigno  los valores antiguos de theta al tiempo 

        for k in range(num_parametros):
            hipotesis=np.matmul(x,theta_temporal)
            error=hipotesis-np.transpose(y)
            derivada=np.dot(error,x[:,k])
            theta[k]=theta_temporal[k]-(alpha/m)*derivada #calculo los nuevos valores de theta 


        #guardamos la funcion costo en cada iteracion    
        J_historia[iteracion]=funcion_costo(x, y,theta) # guardo la funcion costo para cada iteracion, en el historial 

    return theta,J_historia

      

