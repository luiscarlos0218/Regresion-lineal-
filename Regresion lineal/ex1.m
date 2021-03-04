%% Machine Learning Online Class - Exercise 1: Linear Regression

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     warmUpExercise.m

%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
% x refers to the population size in 10,000s
% y refers to the profit in $10,000s
%

%% Initialization
clear ; close all; clc

%% ==================== Part 1: Basic Function ====================
% Complete warmUpExercise.m
fprintf('Running warmUpExercise ... \n');
fprintf('5x5 Identity Matrix: \n');
warmUpExercise()

fprintf('Program paused. Press enter to continue.\n');


pause;


%% ======================= Part 2: Plotting =======================
fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');%Se cargan los datos de las ganancias (Y) y de la población (X)
X = data(:, 1); y = data(:, 2);%separo los datos del txt por columnas 
m = length(y); % number of training examples cantidad de muestras de entrenamiento 


% Plot Data
% Note: You have to complete the code in plotData.m
plotData(X,y); %grafica de los datos 

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Part 3: Cost and Gradient descent ===================

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x %se adiciona una columna de unos para agregar a x0 y poder operar con matrices
theta = zeros(2, 1); % initialize fitting parameters  %se inicializan los parametros teta a cero 

% Some gradient descent settings
iterations = 1500; %numero de iteraciones
alpha = 0.01; %coeficiente de aprendizaje debe ser lo suficientemente pequeño 


fprintf('\nTesting the cost function ...\n')
% compute and display initial cost
J = computeCost(X, y, theta);%se ejecuta la funcion costo 
fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J); %me imprime el valor de la funcion costo con los valores iniciales 0,0 
fprintf('Expected cost value (approx) 32.07\n');%se espera este valor con estos parametros iniciales, solo para verificar la progracacion de la funcion costo 

% further testing of the cost function
J = computeCost(X, y, [-1 ; 2]);% funcion costo evaluado con otros parametros (-1,2)

fprintf('\nWith theta = [-1 ; 2]\nCost computed = %f\n', J);%resultado con los parametros (-1,2)
fprintf('Expected cost value (approx) 54.24\n');%esperados para comprobar programacion 


fprintf('Program paused. Press enter to continue.\n');%una pausa para continuar el programa, fin de seccion 
pause;

fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);%ejecutando la funcion gradiente descendiente 

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);
fprintf('Expected theta values (approx)\n');
fprintf(' -3.6303\n  1.1664\n\n');%para comprobar la programacion 

% Plot the linear fit
hold on; % keep previous plot visible 
plot(X(:,2), X*theta, '-')%grafico la linea de la linealizacion, con los datos de poblacion(X) y los de ganancias predichos por mi modelo 
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;%para cpmprobar la programacion 
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;%para comprobar la programacion 
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

fprintf('Program paused. Press enter to continue.\n');


pause;

%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')% visualizacion de la funcion costo, opcional por ser esta vez dos parametros 

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);%valores de theta 0 para el eje de la grafica 
theta1_vals = linspace(-1, 4, 100);%valores de visualizacion de theta 1 para el eje de la grafica 

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));%matriz de J para cada una de las parejas de thetas, inicializado en 0

% Fill out J_vals  %llenado de los valores de j para cada uno de los thetas
% 
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t); %evaluacion de la funcion costo para cada uno de los pares de thetas 
    end
end


% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)%grafica de superficie 3d 
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);% punto de los thetas encontrados por el modelo sobre la grafica de contorno 







