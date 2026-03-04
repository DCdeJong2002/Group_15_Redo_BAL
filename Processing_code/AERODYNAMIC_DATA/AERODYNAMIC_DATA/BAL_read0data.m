%% Function BAL_read0data.m
% Reads LTT balance zero-measurement data files
% =========================================================================
% Tomas Sinnige - T.Sinnige@tudelft.nl 
% TU Delft - LR - AWEP - Flight Performance and Propulsion
%
% Version: 1.0
% Last updated:  28 April 2020
% First version: 10 October 2017
% =========================================================================
% | Version |    Date   |   Author  |              Changelog              |
% |---------|-----------|-----------|-------------------------------------|
% |   1.0   | 28/04/'20 | T.Sinnige | Changed loop for averaging repeated |
% |         |           |           | measurements. Previously data were  |
% |         |           |           | averaged but the repeated points not|
% |         |           |           | removed from the data array.        |
% |---------|-----------|-----------|-------------------------------------|
% |   0.0   | 10/10/'17 | T.Sinnige | First version                       |
% |---------|-----------|-----------|-------------------------------------|
% =========================================================================
% Inputs:  folder - path to data folder containing file 'fn'
%          fn     - path to data file (excluding the folder component)
%          idxB   - structure containing indices balance data (W3D)
% -------------------------------------------------------------------------
% Outputs: BAL0   - structure containing zero-measurement data
%                     BAL0{i}.raw -> raw data (from measurement file)
%                     BAL0{i}.avg -> raw data but with sequentially 
%                                    repeated measurement points averaged
% =========================================================================
function [BAL0] = BAL_read0data(folder,fn,idxB)

% convert variable fn to cell if not given as cell as input
if ~iscell(fn)
    fn = {fn};
end

% Loop over all files to load the data
for i=1:length(fn) % for loop over the different files
    %% Load Data
    [folder,fn{i}];
    fid  = fopen([folder,fn{i}]);
    DATA{i}.raw= cell2mat(textscan(fid,['%f %f:%f:%f',repmat(' %f',1,10)],'headerlines',2));
    fclose(fid);
    
    %% Check for repeated measurements 
    
    
    % get indices of the repeated datapoints (constant AoA+AoS)
    AoAzero = round(DATA{i}.raw(:,idxB.zero.AoA)*20)/20; % angles of attack [deg] (rounded to 0.05)
    AoSzero = round(DATA{i}.raw(:,idxB.zero.AoS)*20)/20; % angles of attack [deg] (rounded to 0.05)
    angles  = [AoAzero,AoSzero];
       
    [u,I,J] = unique(angles, 'rows', 'first');
%     hasDuplicates = size(u,1) < size(angles,1); % not used anymore

    % average the repeated datapoints and remove from the data
    for j=1:size(u,1)
        DATA{i}.avg(j,:) = mean(DATA{i}.raw(ismember(J,j).',:),1);
    end
%%Organize data in structure way
    BAL0.run=DATA{i}.avg(:,idxB.zero.run);
    BAL0.hr=DATA{i}.avg(:,idxB.zero.hr);
    BAL0.min=DATA{i}.avg(:,idxB.zero.min);
    BAL0.sec=DATA{i}.avg(:,idxB.zero.sec);
    BAL0.AoA=DATA{i}.avg(:,idxB.zero.AoA);
    BAL0.AoS=DATA{i}.avg(:,idxB.zero.AoS);
    BAL0.pBar=DATA{i}.avg(:,idxB.zero.pBar);
    BAL0.temp=DATA{i}.avg(:,idxB.zero.temp);
    BAL0.B1=DATA{i}.avg(:,idxB.zero.B1);
    BAL0.B2=DATA{i}.avg(:,idxB.zero.B2);
    BAL0.B3=DATA{i}.avg(:,idxB.zero.B3);
    BAL0.B4=DATA{i}.avg(:,idxB.zero.B4);
    BAL0.B5=DATA{i}.avg(:,idxB.zero.B5);
    BAL0.B6=DATA{i}.avg(:,idxB.zero.B6);
    
% OLD:
%     BAL0{i}.avg  = BAL0{i}.raw; % initialize average output structure with the original structure
%     if hasDuplicates
%         for j=1:size(u,1)
%             BAL0{i}.avg(j,:) = mean(BAL0{i}.raw(ismember(J,j).',:),1);
%             
%         end % end for loop over repeated datapoints
%     end   
end % end for loop over the zero-measurement data files
end % end of function BAL_read0data.m