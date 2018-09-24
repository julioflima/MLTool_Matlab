function [] = plot_labeled_neurons(PAR)

% --- Plot Neurons Grid with labels ---
%
%   [] = plot_labeled_neurons(PAR)
%
%   Input:
%       PAR.
%           C = prototypes [p x k] or [p x Nlin x Ncol]
%           index = indexes indicating each sample's cluster [1 x N]
%           SSE = squared error of each turn of training [1 x Nep]
%           label = class of each neuron [1 x k] or [1 x Nlin x Ncol]
%   Output:
%       "void" (print a graphic at screen)

%% INITIALIZATIONS

% Get Clusters and its labels
C = PAR.C;
label = PAR.label;

% Get grid dimensions and number of classes
dim = size(C);
classes = length(unique(label));

% Main types of markers, line style and colors

if classes > 7,
    color_array = cell(1,classes+1);
    color_array(1:7) = {'y','m','c','r','g','b','k'};
    color_array(classes+1) = {'w'}; %last one is white
    for i = 8:classes,
        color_array(i) = {rand(1,3)};
    end
else
    color_array = {'y','m','c','r','g','b','k','w'};
end

marker_array = {'.','*','o','x','+','s','d','v','^','<','>','p','h'};

%% ALGORITHM

% Begin Figure
figure;
hold on

% 2D GRID
if dim > 2,
    % Define Axis
    axis ([0 dim(3)+1 0 dim(2)+1]);
    for i = 1:dim(2),
        for j = 1:dim(3),
            % label 0 indicates a non-representative cluster
            if label(i,j) ~= 0,
                plot_color = color_array{label(i,j)};
                line_style =  marker_array{label(i,j)};
                plot (j,i,'Color',plot_color,'LineStyle', line_style);
            end
        end
    end
% 1D GRID
else
    axis ([0 dim(2)+1 -1 +1]);
    for i = 1:dim(2),
        if label(i) ~= 0,
            plot_color = color_array{label(i)};
            line_style =  marker_array{label(i)};
            plot (i,0,'Color',plot_color,'LineStyle', line_style);
        end
    end
end

% Finish Figure
hold off

%% FILL OUTPUT STRUCTURE

% Don't have output structure

%% END