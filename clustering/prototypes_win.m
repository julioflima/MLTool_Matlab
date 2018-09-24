function [c] = prototypes_win(C,sample,PAR)

% --- Calculate the closest prototype to a sample ---
%
%   [c] = prototype_win(C,sample,PAR)
%
%   Input:
%       C = prototypes [p x k] or [p x k1 x k2]
%       sample = data vector [p x 1]
%       PAR.
%           dist = Type of distance 
%               0: dot product
%               2: euclidean distance
%   Output:
%       c = vector with closest prototype position(s)

%% ALGORITHM

% For 1D
if (length(size(C)) == 2),
    
    % Init Variables
    
    [~,k] = size(C);        % Number of prototypes
    Vdist = zeros(1,k);     % Vector of distances
    dist = PAR.dist;        % Type of distance
    win = 0;                % Winner prototype
    
    % Calculate Distantes
    
    for i = 1:k,
        
        % Get Prototype
        prototype = C(:,i);
        
        % dot product
        if(dist == 1)
            Vdist(i) = (sample')*prototype;
            
            % euclidean distance
        elseif (dist == 2)
            Vdist(i) = sum((sample - prototype).^2);
        end
        
    end
    
    % Choose Closest Prototype
    
    % dot product
    if(dist == 0)
        [~,win] = max(Vdist);
        
        % euclidean distance
    elseif (dist == 2)
        [~,win] = min(Vdist);
        
    end

% For 2D
elseif(length(size(C)) == 3),
    
    % Init Variables
    
    [~,k1,k2] = size(C);        % Number of prototypes
    Mdist = zeros(k1,k2);       % Matrix of distances
    dist = PAR.dist;            % Type of distance
    win = [0 0];                % Winner prototype
    
    % Calculate Distantes
    
    for i = 1:k1,
        for j = 1:k2,
            
            % Get Neuron
            neuron = C(:,i,j);
            % dot product
            if(dist == 1)
                Mdist(i,j) = (sample')*neuron;
            % euclidean distance
            elseif (dist == 2)
                Mdist(i,j) = sum((sample - neuron).^2);
            end
            
        end
    end
    
    % Choose Winner Neuron
    
    dmin = Mdist(1,1);
    dmax = Mdist(1,1);
    
    for i = 1:k1,
        for j = 1:k2,
            
            % dot product
            if(dist == 1)
                if Mdist(i,j) >= dmax
                    win = [i j];
                    dmax = Mdist(i,j);
                end
           	% euclidean distance
            elseif (dist == 2)
                if Mdist(i,j) <= dmin
                    win = [i j];
                    dmin = Mdist(i,j);
                end
            end
            
        end
    end
    
end

%% FILL OUTPUT STRUCTURE

c = win;

%% END