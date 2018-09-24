%% BOXPLOT EXAMPLE

% Gera��o de um gr�fico para an�lise estat�stica dos dados
% �ltima Altera��o: 02/01/2014

clear;
clc;

%% Gera gr�fico de caixa

% linha vermelha: Mediana
% Linhas azuis: 25% a 75%
% Linhas pretas: faixa de valores (sem ser outliers)
% Pontos vermelhos: outliers (plodados individualmente)
% Cada coluna � um boxplot

a = 70; b = 85;                         % Valores m�nimos e m�ximos
Mat_boxplot = a + (b-a).*rand(100,3);   % Gera��o de dados aleat�rios
media = mean(Mat_boxplot);              % M�dia dos dados

labels = {'MLP' 'ELM' 'MLM'};           % Nomes de cada coluna

figure; boxplot(Mat_boxplot, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
title('Classificaiton Rate')    % Titulo da figura
ylabel('% Accuracy')            % label eixo y
xlabel('Classifiers')           % label eixo x
axis ([0 4 40 100])             % Eixos da figura

grid on                         % "Grade"/"Malha" (melhorar visualiza��o)

hold on
plot(media,'*k')                % Plotar m�dia no mesmo gr�fico
hold off

%% END