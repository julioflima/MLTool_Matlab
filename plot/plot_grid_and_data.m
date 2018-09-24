function [] = plot_grid_and_data(DATA,OUT_CL,OPTION)

% --- Plot Clusters Grid and Data ---
%
%   [] = plot_grid_and_data(DATA,OUT_CL,OPTION)
%
%   Input:
%       DATA.
%           dados = input matrix [pxN]
%       OUT_CL.
%           C = prototypes [p x k] or [p x Nlin x Ncol]
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
dim = size(OUT_CL.C);

% Get the number of clusters
clusters = unique(index);
n_clusters = length(clusters);

% Main types of markers, line style and colors
color_array =  {'y','m','c','r','g','b','k','w'};
marker_array = {'.','*','o','x','+','s','d','v','^','<','>','p','h'};

%% ALGORITHM

% Begin Figure
figure(fig)
hold on

% Plot clustered data

for i = 1:n_clusters,

    % Define Color
    if i <= 6,
        plot_color = color_array{i};
    else
        plot_color = rand(1,3);
    end
    
    % Define Line Style
    line_style =  marker_array{1};
    
    % Get samples from especific cluster
    samples = find(index == clusters(i));
    
    % Plot samples
    plot(dados(Xaxis,samples),dados(Yaxis,samples), ...
         'Color',plot_color,'LineStyle', line_style)
end

% Plot clusters' prototypes
C = OUT_CL.C;


% 2D cluster grid
if length(dim) > 2,
    % Plot of prototypes' lines
    for i  = 1:dim(2),
       plot(C(Xaxis,:,i),C(Yaxis,:,i),'-ms',...
       'LineWidth',1,...
       'MarkerEdgeColor','k',...
       'MarkerFaceColor','g',...
       'MarkerSize',5)
    end
    % Plot of prototypes' colummns
    for i = 1:dim(3),
       plot(C(Xaxis,i,:),C(Yaxis,i,:),'-ms',...
       'LineWidth',1,...
       'MarkerEdgeColor','k',...
       'MarkerFaceColor','g',...
       'MarkerSize',5)   
    end
% 1D cluster grid
else
	plot(C(Xaxis,:),C(Yaxis,:),'-ms',...
	'LineWidth',1,...
	'MarkerEdgeColor','k',...
	'MarkerFaceColor','g',...
	'MarkerSize',5)
end

% Finish Figure
hold off

%% FILL OUTPUT STRUCTURE

% Don't have output structure

%% END