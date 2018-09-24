function [h] = prototypes_neig (neig,win,Ni,Nn,tmax,t,Vo,Vt)

% --- Prototypes Neighborhood Function ---
%
%   [h] = prototypes_neig (neig,win,Ni,Nn,tmax,t,Vo,Vt)
%
%   Input:
%       neig = type of neighborhood function
%           1:  if winner or neighbor, h = 1, else h = 0.
%           ~1: if neighbor, h = exp (-(||ri -ri*||^2)/(V^2))
%                where: V = Vo*((Vt/Vo)^(t/tmax))
%       win = winner neuron position(s)
%       Ni = prototypes' current neuron position(s)
%       Nn = number of neighbors
%       tmax = max number of iterations
%       t = current iteration
%       Vo = initial value of V
%       Vt = final value of V
%   Output:
%       h = neigborhood function result

%% ALGORITHM

% For 1D
if (length(win) == 1),
    
    if neig == 1,
        if abs(win - Ni) > Nn,
            h = 0;
        else
            h = 1;
        end
    else
        if abs(win - Ni) > Nn,
            h = 0;
        else
            V = Vo*((Vt/Vo)^(t/tmax));
            h = exp(-(win - Ni)^2/(V^2));
        end
    end
    
% For 2D
elseif(length(win) == 2),
    
    if neig == 1,
        if ((abs(win(1) - Ni(1)) > Nn) || (abs(win(2) - Ni(2)) > Nn)),
            h = 0;
        else
            h = 1;
        end
    else
        if ((abs(win(1) - Ni(1)) > Nn) || (abs(win(2) - Ni(2)) > Nn)),
            h = 0;
        else
            V = Vo*((Vt/Vo)^(t/tmax));
            h = exp(-((win(1) - Ni(1))^2+(win(2) - Ni(2))^2)/(V^2));
        end
    end
    
end

%% END