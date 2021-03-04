# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 21:33:58 2021

@author: Luis Carlos Prada 
Ejercicio de regresion lineal machine learning.Codigo  principal
"""



"Importamos todas las librerias"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from funcion_costo import funcion_costo
from gradiente_descendiente import gradiente_descendiente


#cargamos los datos del problema 
datos=pd.read_csv("ex1data1.txt",header=None)
datos_2=datos.to_numpy()
x=np.mat(datos_2[:,0])
y=np.mat(datos_2[:,1])
y=np.transpose(y)
x=np.transpose(x)
m=x.shape[0]

"***___Graficar los datos_______**" 
plt.plot(x,y,'x',color='r',label='Datos de entrenamiento')
plt.xlabel('Poblacion de la ciudad en 10000s')
plt.ylabel('Ganancias en $10000s')


"**______Gradiente descendiente________**"
unos=np.ones([m,1])
x=np.concatenate((unos,x),axis=1) #adicionar una columna de unos para los calculos x0=1

#Parametros del algoritmo 
iteraciones=1500 
alpha=0.01 #coeficiente de aprendizaje debe ser lo suficientemente pequeño 
theta=np.zeros([2,1])

#Funcion costo 
J=funcion_costo(x,y,theta)#se ejecuta la funcion costo 
print("Probando la funcion costo...")
print(f"con tetha[0,0] el valor de la funcion costo es {J} \n")
J=funcion_costo(x,y,np.mat([[-1],[2]]))
print(f"con tetha[-1,2] el valor de la funcion costo es {J} \n")

#Funcion gradiente descendiente
print("Corriendo gradiente descendiente...")
theta=gradiente_descendiente(x,y,theta,alpha,iteraciones)[0]
print(f'Los Parametros encontradoa para el gradiente descendiente son {theta}')

#Grafica del modelo lienal ajustada a los datos 

plt.plot(x[:,1],np.matmul(x,theta),label='Regresion lineal')#grafico la linea de la linealizacion, con los datos de poblacion(X) y los de ganancias predichos por mi modelo 
plt.legend()
plt.show()


# Ejemplo para predecir valores para tamaños de poblacion de 35000 y 70000
prediccion_1 = np.matmul(np.mat([1, 3.5]),theta) #para comprobar la programacion 
print(f'Para una poblacion = 35,000, se predice una ganancia de {prediccion_1*10000}')
prediccion_2 = np.matmul(np.mat([1, 7]),theta) #para comprobar la programacion 
print(f'Para una poblacion = 70,000, se predice una ganancia de {prediccion_2*10000}')

"***________Visualizacion de J y theta_________***"

#Valores sobre los cuales se calcula J
valores_theta0 = np.linspace(-10, 10, 100) #valores de theta 0 para el eje de la grafica 
valores_theta1 = np.linspace(-1, 4, 100) #valores de visualizacion de theta 1 para el eje de la grafica 

#inicializar los valores de la matriz de J a 0 
valores_J= np.zeros([len(valores_theta0),len(valores_theta1)]) #matriz de J para cada una de las parejas de thetas, inicializado en 0

# llenado de los valores de j para cada uno de los thetas
for i in  range(len(valores_theta0)):
    for j in range(len(valores_theta1)):
        t= np.mat([[valores_theta0[i]],[valores_theta1[j]]])
        valores_J[i,j] = funcion_costo(x, y, t) #evaluacion de la funcion costo para cada uno de los pares de thetas 


# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
#J_vals = J_vals'

#Figura 3D
#crear la figura
fig = plt.figure()

# Tomo el eje actual y defino una proyección 3D
ax= fig.add_subplot(projection='3d')
valores_J=np.transpose(valores_J)

# Grafico surface en 3D
surface = ax.plot_surface(valores_theta0,valores_theta1, valores_J,rstride=1, cstride=1, cmap='coolwarm', linewidth=0)
plt.xlabel('theta_0') 
plt.ylabel('theta_1')
plt.show()


# Grafica de contorno 
plt.contour(valores_theta0,valores_theta1, valores_J, levels=30)
plt.xlabel('theta_0') 
plt.ylabel('theta_1')
plt.plot(theta[0], theta[1], 'x', color='r',markersize=12) #punto de los thetas encontrados por el modelo sobre la grafica de contorno 


