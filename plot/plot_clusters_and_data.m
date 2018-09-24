function [] = plot_clusters_and_data(DATA,OUT_CL,OPTION)

% --- Plot clusters Organization ---
%
%   [] = plot_clusters(DATA,OUT_CL,OPTION)
%
%   Input:
%       DATA.
%           dados = input matrix [pxN]
%       OUT_CL.
%           C = prototypes [p x k]
%           index = indexes indicating each sample's cluster [1 x N]
%           SSE = squared error of each turn of training [1 x Nep]
%       OPTION.
%           Xaxis = Attribute to be plotted at x axis
%           Yaxis = Attribute to be plotted at y axis
%           fig = Define figure number
%   Output:
%       "void" (print a graphic at screen)

%% INITIALIZATIONS

% Get figure options
Xaxis = OPTION.Xaxis;
Yaxis = OPTION.Yaxis;
fig = OPTION.fig;

% Get Data
dados = DATA.input;

% Get index and Clusters' Dimensions
index = OUT_CL.index;
C = OUT_CL.C;
dim = size(C);

% Get the number of clusters
clusters = unique(index);
n_clusters = length(clusters);

% Main types of colors and markers
color_array =  {'y','m','c','r','g','b','k','w'};
marker_array = {'.','*','o','x','+','s','d','v','^','<','>','p','h'};

%% ALGORITHM

% Begin Figure
figure(fig)
hold on

% Define figure properties
s1 = 'Attribute ';  s2 = int2str(Xaxis);    s3 = int2str(Yaxis);
xlabel(strcat(s1,s2));
ylabel(strcat(s1,s3));
title ('2D of Clusters Distribution');

% Plot Data

for i = 1:n_clusters,

    % Define Color
    if i <= 6,
        plot_color = color_array{i};
    else
        plot_color = rand(1,3);
    end
    
    % Define Marker as the LineStyle
    marker = marker_array{1};

    % Get samples from especific cluster
    samples = find(index == clusters(i));
    
    % Plot samples
    plot(dados(Xaxis,samples),dados(Yaxis,samples),marker, ...
        'MarkerFaceColor',plot_color)
end

% Plot clusters' prototypes

% Verify if it is a 2D cluster grid, and adjust it
if length(dim) > 2,
    C = zeros(dim(1),dim(2)*dim(3));
    aux = 0;
    for i = 1:dim(2),
        for j = 1:dim(3),
            aux = aux + 1;
            C(:,aux) = OUT_CL.C(:,i,j);
        end
    end
% 1D cluster grid - Dont need adjustment
else
    C = OUT_CL.C;
end

plot(C(Xaxis,:),C(Yaxis,:),'k*')

% Finish Figure
hold off

%% FILL OUTPUT STRUCTURE

% Don't have output structure

%% END