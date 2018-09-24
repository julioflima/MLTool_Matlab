function [DATAout] = data_class_loading(OPTION)

% --- Selects a Data Base ---
%
%   [DATAout] = data_loading(OPTION)
% 
%   Input:
%       OPTION.prob = which data base will be used
%           01: boxes
%           02: dermatology
%           03: faces
%           04: four groups
%           05: images data
%           06: iris data
%           07: motor short circuit failure
%           08: motor short circuit filtered
%           09: random
%           10: Spine
%           11: Two Moons
%           12: Wine
%           13: Motor broken bar failure multi class
%           14: Motor broken bar failuer binary class
%           15: Breast Cancer
%           16: Cryotherapy
%           17: Immunotherapy
%       OPTION.prob2 = specify data set
%           uses this field: 07, 08 
%   Output:
%       DATA.
%           input = input matrix [p x N]
%           output = output matrix [1 x N] (sequential: 1, 2...)
%           rot = mantain original labels

%% INITIALIZATION

DATA = struct('input',[],'output',[],'lbl',[]);

choice = OPTION.prob;

%% ALGORITHM

switch (choice),
    
    case 1,
        % Load Boxes Data
        loaded_data1 = load('data_boxes_train.dat');
        loaded_data2 = load('data_boxes_class.dat');
        DATA.input = [loaded_data1(:,1:end-1)' loaded_data2(:,1:end-1)'];  
        DATA.output = [loaded_data1(:,end)' loaded_data2(:,end)'];
        DATA.lbl = DATA.output;                  % Original Labels
    case 2,
        % Load Dermatology Data
        loaded_data = importdata('data_dermato_03.txt');
        DATA.input = loaded_data(:,1:end-1)';	% Input
        DATA.output = loaded_data(:,end)';    	% Output
        DATA.lbl = DATA.output;               	% Original Labels
    case 3,
        % Load Faces Data (YALE)
        disp('Still Not implemented. Void Structure Created')
    case 4,
        % Load Four Groups Data
        load data_four_groups.mat;
        DATAaux.input = DATA.dados;
        DATAaux.output = DATA.alvos;
        DATAaux.lbl = DATA.rot;
        DATA = DATAaux;
    case 5,
        % Load Images Data
        disp('Still Not implemented. Void Structure Created')
    case 6,
        % Load Iris Data
        loaded_data = importdata('data_iris.m');
        DATA.input = loaded_data(:,1:end-1)';     % Input
        DATA.output = loaded_data(:,end)';         % Output
        DATA.lbl = DATA.output;                    % Rotulos
    case 7,
        % Load Motor Failure Data
        if (isfield(OPTION,'prob2'))
            DATA = data_motor_gen(OPTION);
        else
            disp('Specify the database')
        end
    case 8,
        % Load Motor Failure filtered Data
        if(isfield(OPTION,'prob2')),
            DATA = data_motor_filt_gen(OPTION);
        else
            disp('Specify the database')
        end
    case 9,
        % Load Random Data
        DATA.input = rand(2,294);
        DATA.output = ceil(3*rand(1,294));
        DATA.lbl = DATA.output;
    case 10,
        % Load Spine Data
        spine = importdata('data_spine.mat');
        DATA.input = spine(:,1:6)';
        DATA.output = spine(:,7:9)';
        DATA.lbl = DATA.output;
        OPT.lbl = 3;
        DATA = label_adjust(DATA,OPT);
    case 11,
        % Load Two Moons Data
        loaded_data = load('data_two_moons.dat');
        DATA.input = loaded_data(:,1:2)';
        loaded_data(502:end,3) = 2;
        DATA.output = loaded_data(:,3)';
        DATA.lbl = DATA.output;
    case 12,
        % Load Wine Data
        loaded_data = importdata('data_wine_03.m');
        DATA.input = loaded_data(:,1:end-1)';	% Input
        DATA.output = loaded_data(:,end)';   	% Output
        DATA.lbl = DATA.output;                  % Rotulos
    case 13,
        % Load Motor broken bar multi class data
        DATA_aux = load('data_bb.mat');
        DATA.input = DATA_aux.DATA.input;
        DATA.output = DATA_aux.DATA.output;
        DATA.lbl = DATA.output;
    case 14,
        % Load Motor broken bar binary class data
        DATA_aux = load('data_bb.mat');
        DATA.input = DATA_aux.DATA.input;
        output = DATA_aux.DATA.output;
        DATA.lbl = output;
        for i = 1:2520,
           if output(i) ~= 1,
               output(i) = 2;
           end
        end
        DATA.output = output;
    case 15,
        loaded_data = importdata('data_breast_cancer.txt');
        data_aux = zeros(683,11);
        cont = 0;
        for i = 1:699,
            if (loaded_data(i,8) ~= 0),
                cont = cont + 1;
                data_aux(cont,:) = loaded_data(i,:);
            end
        end
        DATA.input = data_aux(:,3:11)';
        DATA.output = data_aux(:,2)'/2;
        DATA.lbl = DATA.output;
    case 16,
        loaded_data = importdata('data_cryotherapy.txt');
        DATA.input = loaded_data(:,1:6)';
        DATA.output = loaded_data(:,7)'+1;
       	DATA.lbl = DATA.output;
    case 17,
        loaded_data = importdata('data_immunotherapy.txt');
        DATA.input = loaded_data(:,1:7)';
        DATA.output = loaded_data(:,8)'+1;
       	DATA.lbl = DATA.output;
    otherwise
        % None of the sets
        disp('Unknown Data Base. Void Structure Created')
end

%% FILL OUTPUT STRUCTURE

DATAout = DATA;

%% END