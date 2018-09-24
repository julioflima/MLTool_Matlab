function [frame] = prototypes_frame(C,DATA)

% --- Save current frame for clustering function ---
%
%   [frame] = save_frame(C,DATA)
%
%   Input:
%       C = prototypes [p x k] or [p x k(1) x k(2)]
%       DATA.
%           input = input matrix [p x N]
%   Output:
%       frame = struct containing 'cdata' and 'colormap'

%% INITIALIZATION

input = DATA.input;

%% ALGORITHM

% Convert 2d Prototypes to 1d Prototypes
if (length(size(C)) == 3),
    C = permute(C,[2 3 1]);
    C = reshape(C,[],size(C,3),1);
    C = C';
end

% Plot Clusters
cla;
plot(C(1,:),C(2,:),'k*')
hold on
plot(input(1,:),input(2,:),'r.')
hold off
frame = getframe;

%% END