# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 13:50:08 2021

@author:Luis Carlos Prada
Regresion lineal para multiples variables 

"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from normalizacion_datos import normalizacion_datos
from gradiente_descendiente import gradiente_descendiente

'''***_____Cargar archivos_______***'''

datos=pd.read_csv('ex1data2.txt', header=None)
x=datos[[0,1]]
y=datos.get(2)
m=len(y.values)

[x_norm,mu,sigma]=normalizacion_datos(x)

unos=np.ones([m,1])
x_norm=np.concatenate((unos,x_norm.values),axis=1)
x_norm=pd.DataFrame(x_norm)

'''**________Valores iniciales gradiente descendiente________**'''

alpha = 0.1
num_iters = 50
numero_tethas=x_norm.shape[1]
tetha_inicial=np.zeros([numero_tethas,1])
[tetha,J_historia]=gradiente_descendiente(x_norm.values,y.values,tetha_inicial,alpha,num_iters)

plt.plot(range(len(J_historia)), J_historia);
plt.xlabel('Numero de iteraciones');
plt.ylabel('Costo J')
