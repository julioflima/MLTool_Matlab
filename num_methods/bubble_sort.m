function [OUT] = bubble_sort(in_vec,in_order)

% --- Bubble sorting algorithm ---
%
%   [OUT] = bubble_sort(in_vec,in_order)
%
%   Input:
%       in_vec = vector to be organized
%       in_order = encreasing (1) or decreasing order (2)
%   Output:
%       OUT.
%           out_vec = organized vector
%           out_order = holds order of original data

%% INITIALIZATIONS

% Get vector size
vec_size = length(in_vec);

% Init Variables
out_order = (1:vec_size)';

%% ALGORITHM

% Encreasing order
if in_order == 1,
    for i = 1:(vec_size - 1),
        % Restore stop flag
        stop_flag = 1;
        % do bubbling
        for j = 1:(vec_size - 1),
            if (in_vec(j) > in_vec(j+1)),
                % adjust input vector
                aux = in_vec(j);
                in_vec(j) = in_vec(j+1);
                in_vec(j+1) = aux;
                % adjust output order
                aux = out_order(j);
                out_order(j) = out_order(j+1);
                out_order(j+1) = aux;
                % clear stop flag
                stop_flag = 0;
            end
        end
        % breaks if stop flag is set 
        if stop_flag == 1,
            break;
        end
    end
% Decreasing order
else
    for i = 1:(vec_size - 1),
        % Restore stop flag
        stop_flag = 1;
        % do bubbling
        for j = 1:(vec_size - 1),
            if (in_vec(j) < in_vec(j+1)),
                % adjust input vector
                aux = in_vec(j);
                in_vec(j) = in_vec(j+1);
                in_vec(j+1) = aux;
                % adjust output order
                aux = out_order(j);
                out_order(j) = out_order(j+1);
                out_order(j+1) = aux;
                % clear stop flag
                stop_flag = 0;
            end
        end
        % breaks if stop flag is set 
        if stop_flag == 1,
            break;
        end
    end
end

%% FILL OUTPUT STRUCTURE

OUT.out_order = out_order;
OUT.out_vec = in_vec;

%% END