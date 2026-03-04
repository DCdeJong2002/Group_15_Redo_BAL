%% Function BAL_zero.m
% Subtracts zero offset from balance forces (LTT)
% =========================================================================
% Tomas Sinnige - T.Sinnige@tudelft.nl 
% TU Delft - LR - AWEP - Flight Performance and Propulsion
%
% Version: 1.1
% Last updated:  28 April 2020
% First version: 17 October 2017
% =========================================================================
% | Version |    Date   |   Author  |              Changelog              |
% |---------|-----------|-----------|-------------------------------------|
% |   1.1   | 28/04/'20 | T.Sinnige | -) Using incidence angles from zero |
% |         |           |           | run(s) now to define whether we're  |
% |         |           |           | dealing with an alfa or beta sweep. |
% |         |           |           | -) Added check on match between     |
% |         |           |           | angles in measurement file and zero |
% |         |           |           | data file                           |
% |---------|-----------|-----------|-------------------------------------|
% |   1.0   | 28/04/'20 | T.Sinnige | Added functionality for cases with  |
% |         |           |           | single incidence angle in zero data.|
% |---------|-----------|-----------|-------------------------------------|
% |   0.0   | 17/10/'17 | T.Sinnige | First version (split from           |
% |         |           |           |                BAL_forces.m)        |
% |---------|-----------|-----------|-------------------------------------|
% =========================================================================
% Inputs:  BAL      - structure containing measurement data balance
%          BAL0     - structure containing zero-measurement data balance
%          idxB     - structure containing indices balance data (W3D)
% -------------------------------------------------------------------------
% Outputs: BAL      - structure containing measurement data balance
%                     (with processed forces and moments added)
%                       BAL.raw       -> raw data (from raw_.. file)
%                       BAL.unc       -> uncorrected data (from unc_.. file)
%                       BAL.cor       -> corrected data (from cor_.. file)
%                       BAL.BAL0      -> zero-measurement data
%                       BAL.B16zeroed -> raw data of B1-6 (steps), with
%                                        zero offset removed
% =========================================================================
function BAL = BAL_zero(BAL,BAL0,idxB)
 
%% combine zero-measurement data
BAL0tot = [];    
DATA=[BAL0.run,BAL0.hr,BAL0.min, BAL0.sec,BAL0.AoA,BAL0.AoS,BAL0.pBar,BAL0.temp,BAL0.B1,BAL0.B2,BAL0.B3,BAL0.B4,BAL0.B5,BAL0.B6];
BAL0tot = cat(3,BAL0tot,DATA);
 
%% Check whether AoA or AoS was varied, or both (latter case not implemented yet)
% AoA0 = arrayfun(@(i) round(BAL.BAL0.raw{i}.raw(:,idxB.zero.AoA)*20)/20,1:length(BAL.BAL0.raw),'UniformOutput',false);
% AoS0 = arrayfun(@(i) round(BAL.BAL0.raw{i}.raw(:,idxB.zero.AoS)*20)/20,1:length(BAL.BAL0.raw),'UniformOutput',false);
AoA0 = arrayfun(@(i) round(BAL0.AoA*20)/20,1,'UniformOutput',false);
AoS0 = arrayfun(@(i) round(BAL0.AoS*20)/20,1,'UniformOutput',false);

AoA0unique = unique(cell2mat(arrayfun(@(i) AoA0{i}.',1:length(AoA0),'UniformOutput',false)));
AoS0unique = unique(cell2mat(arrayfun(@(i) AoS0{i}.',1:length(AoS0),'UniformOutput',false)));

if length(AoA0unique)>=1 && length(AoS0unique)==1
    attMode = 'AoA';
    angle0 = AoA0;
    angleOther = AoS0;
    angle0meas = round(BAL.AoA*20)/20; % angle(s) of attack of measurements
    angleOtherMeas = round(BAL.AoS*20)/20; % angle(s) of sideslip    
elseif length(AoS0unique)>1 && length(AoA0unique)==1 
   attMode = 'AoS';
   angle0 = AoS0;  
   angleOther = AoA0;
   angle0meas = round(BAL.AoS*20)/20; % angle(s) of sideslip of measurements
   angleOtherMeas = round(BAL.AoA*20)/20; % angle(s) of attack
else
   attMode = 'both';
   angle0 = AoA0;
   angleOther = AoS0;
   angle0meas = round(BAL.AoA*20)/20; % angle(s) of sideslip of measurements
   angleOtherMeas = round(BAL.AoS*20)/20; % angle(s) of attack
   display('Both AoA and AoS varied in same zero run. Program will only work if single zero run is used per data file. Polynomial regression used to define 0 offsets. Check response model with raw zero data.')   
%    error('Both AoA and AoS varied in same zero run. Program cannot deal with this .')
end

% if length(unique(round(BAL.raw(:,idxB.raw.AoA)*20)/20)) >= 1 && ...
%    length(unique(round(BAL.raw(:,idxB.raw.AoS)*20)/20)) == 1  
%    attMode = 'AoA';
%    angle0 = AoA0;  
%    angleOther = AoS0;
%    angle0meas = round(BAL.raw(:,idxB.raw.AoA)*20)/20; % angle(s) of attack of measurements
%    angleOtherMeas = round(BAL.raw(:,idxB.raw.AoS)*20)/20; % angle(s) of sideslip
% elseif length(unique(round(BAL.raw(:,idxB.raw.AoA)*20)/20)) == 1 && ...
%    length(unique(round(BAL.raw(:,idxB.raw.AoS)*20)/20)) > 1  
%    attMode = 'AoS';
%    angle0 = AoS0;  
%    angleOther = AoA0;
%    angle0meas = round(BAL.raw(:,idxB.raw.AoS)*20)/20; % angle(s) of sideslip of measurements
%    angleOtherMeas = round(BAL.raw(:,idxB.raw.AoA)*20)/20; % angle(s) of attack
% else
%    attMode = 'both';
% %    error('Both AoA and AoS varied in same run. Program cannot deal with this yet.')
% end

%% check whether the angles of incidence in the zero-measurement data are the same in all files
for i=1:1
    if i>1 && (~all(AoA0{i}==AoA0{1}) || ~all(AoS0{i}==AoS0{1}))
        error('Available zero measurements contain data at different angle(s) of attack. Check the input files and/or change zeroMode variable.')    
    end
end

%% get time at which measurements and zero measurements were performed
      
% get the times of the datapoints that are to be zeroed
hrMinSec = [BAL.hr,BAL.min,BAL.sec];
yrMonDay = zeros(size(hrMinSec)); % set all to 0,0,0
yrMonDay(hrMinSec(:,1)<6,3) = 1;
tMeas = datenum([yrMonDay,hrMinSec]);       
        
% compute time of zero measurements
for i=1:size(BAL0tot,1) % loop over the different AoAs

    % read the hr/min/sec of the zero measurements performed at this 
    % angle from the data file
    hrMinSec0 = permute([BAL0tot(i,idxB.zero.hr,:),BAL0tot(i,idxB.zero.min,:),BAL0tot(i,idxB.zero.sec,:)],[3 2 1]);

    % define year/month/day -> measurement data are always taken on the 
    % same calendar day (so y/m/d is set to 0,0,0), except when 
    % datapoints are taken after midnight, then y/m/d = 0,0,1
    yrMonDay0 = zeros(size(hrMinSec0)); % set all to 0,0,0
    yrMonDay0(hrMinSec0(:,1)<6,3) = 1; % correct for datapoints taken after midnight (which is considered here as before 6 am)
    t0 = datenum([yrMonDay0,hrMinSec0]);
end

%% Interpolate in zero offset data

% to-do: make this more reliable
% we may need to take out some parts here to make it simpler


if length(angle0)==1 && length(unique(angle0{1}))==1 % use only 1 zero file per measurement, with all angles the same
    
    % check whether the incidence angle of the measurements is indeed
    % available in the zero measurement
    if ~ (all(ismember(round(angle0meas*20)/20,round(angle0{1}*20)/20)) && all(ismember(round(angleOtherMeas*20)/20,round(angleOther{1}*20)/20)))
%     if ~all(round(unique(angle0{1})*20)/20 == angle0meas) || ~all(round(unique(angleOther{1})*20)/20 == angleOtherMeas)
        error('Angle of incidence of zero measurement is different than the one of the measurement files.')   
    end
    % check whether the other angle is also ok
    
    
    B1_0 = mean(BAL0tot(:,idxB.zero.B1,1));
    B2_0 = mean(BAL0tot(:,idxB.zero.B2,1));
    B3_0 = mean(BAL0tot(:,idxB.zero.B3,1));
    B4_0 = mean(BAL0tot(:,idxB.zero.B4,1));
    B5_0 = mean(BAL0tot(:,idxB.zero.B5,1));
    B6_0 = mean(BAL0tot(:,idxB.zero.B6,1));
elseif length(angle0)==1 && length(unique(angle0{1}))>1 % use only 1 zero file per measurement, and different angles in that file
    % check whether the constant angle is at the same value in the zero run
    % and in the measurement data
    if ~all(ismember(round(angleOtherMeas*20)/20,round(angleOther{1}*20)/20))
        error('Mismatch in angle of incidence between zero file and measurement data. Use other zero file.')
    end
        
   if ~strcmpi(attMode,'both') % single angle varied
       B1_0 = interp1(angle0{1},BAL0tot(:,idxB.zero.B1,1),angle0meas,'pchip',NaN); % this code was already included in the previous version too, hasn't changed.
       B2_0 = interp1(angle0{1},BAL0tot(:,idxB.zero.B2,1),angle0meas,'pchip',NaN);
       B3_0 = interp1(angle0{1},BAL0tot(:,idxB.zero.B3,1),angle0meas,'pchip',NaN);
       B4_0 = interp1(angle0{1},BAL0tot(:,idxB.zero.B4,1),angle0meas,'pchip',NaN);
       B5_0 = interp1(angle0{1},BAL0tot(:,idxB.zero.B5,1),angle0meas,'pchip',NaN);
       B6_0 = interp1(angle0{1},BAL0tot(:,idxB.zero.B6,1),angle0meas,'pchip',NaN); 
   else % both angles varied
       
       
       
       for j=1:length(angle0meas)
           idx = angle0{1}==angle0meas(j) & angleOther{1}==angleOtherMeas(j);
           if sum(idx)==0 % measured angles not within range of the wind-off measurement! 
               B1_0(j,1) = NaN;
               B2_0(j,1) = NaN;
               B3_0(j,1) = NaN;
               B4_0(j,1) = NaN;
               B5_0(j,1) = NaN;
               B6_0(j,1) = NaN;
          else
               B1_0(j,1) = mean(BAL0tot(idx,idxB.zero.B1,1));
               B2_0(j,1) = mean(BAL0tot(idx,idxB.zero.B2,1));
               B3_0(j,1) = mean(BAL0tot(idx,idxB.zero.B3,1));
               B4_0(j,1) = mean(BAL0tot(idx,idxB.zero.B4,1));
               B5_0(j,1) = mean(BAL0tot(idx,idxB.zero.B5,1));
               B6_0(j,1) = mean(BAL0tot(idx,idxB.zero.B6,1));
           end
           
       end
       
           
       
       
       if sum(isnan([B1_0,B2_0,B3_0,B4_0,B5_0,B6_0]))>0
           %% fitting approach
           B1_0_fit = fit([angle0{1} angleOther{1}],BAL0tot(:,idxB.zero.B1,1),'poly22');
           B2_0_fit = fit([angle0{1} angleOther{1}],BAL0tot(:,idxB.zero.B2,1),'poly22');
           B3_0_fit = fit([angle0{1} angleOther{1}],BAL0tot(:,idxB.zero.B3,1),'poly22');
           B4_0_fit = fit([angle0{1} angleOther{1}],BAL0tot(:,idxB.zero.B4,1),'poly22');
           B5_0_fit = fit([angle0{1} angleOther{1}],BAL0tot(:,idxB.zero.B5,1),'poly22');
           B6_0_fit = fit([angle0{1} angleOther{1}],BAL0tot(:,idxB.zero.B6,1),'poly22');
    
           B1_0 = B1_0_fit([angle0meas,angleOtherMeas]);
           B2_0 = B2_0_fit([angle0meas,angleOtherMeas]);
           B3_0 = B3_0_fit([angle0meas,angleOtherMeas]);
           B4_0 = B4_0_fit([angle0meas,angleOtherMeas]);
           B5_0 = B5_0_fit([angle0meas,angleOtherMeas]);
           B6_0 = B6_0_fit([angle0meas,angleOtherMeas]);
       end

       
   end
   
    
    
    
    
    
    
    
    
elseif length(angle0)>1 % multiple zero files
    % prepare meshgrids with angle and time
    [a,t] = meshgrid(angle0{1},t0);
    
    if strcmpi(attMode,'AoA') || strcmpi(attMode,'AoS')
%         if length(t0)>1 % interpolate in time
            B1_0 = interp2(a,t,permute(BAL0tot(:,idxB.zero.B1,:),[3 1 2]),angle0meas,tMeas);
            B2_0 = interp2(a,t,permute(BAL0tot(:,idxB.zero.B2,:),[3 1 2]),angle0meas,tMeas);
            B3_0 = interp2(a,t,permute(BAL0tot(:,idxB.zero.B3,:),[3 1 2]),angle0meas,tMeas);
            B4_0 = interp2(a,t,permute(BAL0tot(:,idxB.zero.B4,:),[3 1 2]),angle0meas,tMeas);
            B5_0 = interp2(a,t,permute(BAL0tot(:,idxB.zero.B5,:),[3 1 2]),angle0meas,tMeas);
            B6_0 = interp2(a,t,permute(BAL0tot(:,idxB.zero.B6,:),[3 1 2]),angle0meas,tMeas);
%         else
%             B1_0 = interp1(a,BAL0tot(:,idxB.zero.B1,:),angle0meas);
%             B2_0 = interp1(a,BAL0tot(:,idxB.zero.B2,:),angle0meas);
%             B3_0 = interp1(a,BAL0tot(:,idxB.zero.B3,:),angle0meas);
%             B4_0 = interp1(a,BAL0tot(:,idxB.zero.B4,:),angle0meas);
%             B5_0 = interp1(a,BAL0tot(:,idxB.zero.B5,:),angle0meas);
%             B6_0 = interp1(a,BAL0tot(:,idxB.zero.B6,:),angle0meas);
%         end
    elseif strcmpi(attMode,'both')
        error('AoA and AoS varied in mulitple zero runs -> not implemented yet.')  
    end
end

BAL0.intp = [B1_0,B2_0,B3_0,B4_0,B5_0,B6_0];

% subtract zero measurement data from the measured data
BAL.B16zeroed = BAL.B-BAL0.intp;

    
end

% OLD:
% if strcmpi(zeroMode,'single') % zero measurement at single AoA
% 
% if strcmpi(zeroMode,'single') % zero measurement at single AoA
%     
%     % check whether the angles of attack in the zero-measurement data are the same in all files
%     for i=1:length(BAL.BAL0.raw)
%         AoA0{i} = round(BAL.BAL0.raw{i}.avg(:,idxB.zero.AoA)*20)/20; % unique angle(s) of attack of zero measurements     
%         if i>1 && ~all(AoA0{i}==AoA0{2})
%             error('Available zero measurements contain data at different angle(s) of attack. Check the input files and/or change zeroMode variable.')    
%         end
%     end        
%         
%     % check whether the angle of attack of the zero measurement matches
%     % with the angle of attack of the measurement data (to within 0.05 deg)
%     AoAmeas = round(BAL.raw(:,idxB.raw.AoA)*20)/20; % unique angle(s) of attack of measurements
%     if ~all(ismember(unique(AoAmeas),unique(AoA0{1})))
%         error('Zero measurement does not contain data at (some) angle(s) of attack contained in the measurement files. Check the input files and/or change zeroMode variable.')
%     end
% 
% %     % perform interpolation if multiple zero measurements are available
% %     if length(BAL.BAL0.raw)>1 % multiple zero measurements are available
% %         % combine zero-measurement data
% %         BAL0tot = [];
% %         for i=1:length(BAL.BAL0.raw)        
% %             BAL0tot = cat(3,BAL0tot,BAL.BAL0.raw{i}.avg);
% %         end    
% % 
% %         % get time of measurement datapoints
% %         % define year/month/day -> measurement data are always taken on the same
% %         % calendar day (so y/m/d is set to 0,0,0), except when datapoints are 
% %         % taken after midnight, then y/m/d = 0,0,1
% %         hrMinSec = [BAL.raw(:,idxB.raw.hr),BAL.raw(:,idxB.raw.min),BAL.raw(:,idxB.raw.sec)]; % hr/min/sec
% %         yrMonDay = zeros(size(hrMinSec)); % yr/month/day -> set all to 0,0,0
% %         yrMonDay(hrMinSec(:,1)<6,3) = 1; % correct day in case measurement was taken after midnight
% %         tMeas = datenum([yrMonDay,hrMinSec]);
% % %         tMeas = datenum(0,0,0,BAL.raw(:,idxB.raw.hr),BAL.raw(:,idxB.raw.min),BAL.raw(:,idxB.raw.sec));
% % 
% %         % get time of zero-measurement datapoints
% %         hrMinSec0 = permute([BAL0tot(:,idxB.zero.hr,:),BAL0tot(:,idxB.zero.min,:),BAL0tot(:,idxB.zero.sec,:)],[3 2 1]); % hr/min/sec
% %         yrMonDay0 = zeros(size(hrMinSec0)); % yr/month/day -> set all to 0,0,0
% %         yrMonDay0(hrMinSec0(:,1)<6,3) = 1; % correct day in case measurement was taken after midnight
% %         t0 = datenum([yrMonDay0,hrMinSec0]);
% % 
% %         % compute interpolated zero-offset (interpolated between all
% %         % available zero measurement points)
% %         BAL.BAL0.intp = interp1(t0,permute(BAL0tot(:,idxB.zero.B,:),[3 2 1]),tMeas,'linear');
% % 
% %     else % only one zero measurement is available
% %         BAL.BAL0.intp = BAL.BAL0.raw{i}.avg(:,idxB.zero.B);
% %     end
%         
%     BAL.BAL0.intp = 0; 
%     display('Zero loading offsets all set to 0 -> assumes that balance was reset before measurement.')
%     
%     % subtract zero measurement data from the measured data
%     BAL.B16zeroed = BAL.raw(:,idxB.raw.B)-BAL.BAL0.intp;
%     
% elseif strcmpi(zeroMode,'sweep')    
%     
%     % check whether the angles of attack in the zero-measurement data are
%     % the same in all files
%     for i=1:length(BAL.BAL0.raw)
%         AoA0{i} = round(BAL.BAL0.raw{i}.avg(:,idxB.zero.AoA)*20)/20; % unique angle(s) of attack of zero measurements     
%         if i>1 && ~all(AoA0{i}==AoA0{2})
%             error('Available zero measurements contain data at different angle(s) of attack. Check the input files and/or change zeroMode variable.')    
%         end
%     end
% 
%     % check whether the angle of attack sweep of the zero measurement
%     % matches with the angle of attack sweep of the measurement data 
%     % (to within 0.05 deg)
%     AoAmeas = round(BAL.raw(:,idxB.raw.AoA)*20)/20; % unique angle(s) of attack of measurements
%     if ~all(AoAmeas==AoA0{1})
%         error('Zero measurement does not contain the same angle of attack sweep as the measurement data. Check the input files and/or change zeroMode variable.')
%     end
%     
%     % perform interpolation if multiple zero measurements are available
%     if length(BAL.BAL0.raw)>1 % multiple zero measurements are available
%         % combine zero-measurement data
%         BAL0tot = [];
%         for i=1:length(BAL.BAL0.raw)        
%             BAL0tot = cat(3,BAL0tot,BAL.BAL0.raw{i}.avg);
%         end        
%         
%         % get the time of the current measurement
%         hrMinSec = [BAL.raw(:,idxB.raw.hr,:),BAL.raw(:,idxB.raw.min,:),BAL.raw(:,idxB.raw.sec,:)];
%         yrMonDay = zeros(size(hrMinSec)); % set all to 0,0,0
%         yrMonDay(hrMinSec(:,1)<6,3) = 1;
%         tMeas = datenum([yrMonDay,hrMinSec]);       
%         
%         % compute interpolated (in time) zero-measurement data
%         for i=1:size(BAL0tot,1) % loop over the different AoAs
% 
%             % read the hr/min/sec of the zero measurements performed at this 
%             % angle from the data file
%             hrMinSec0 = permute([BAL0tot(i,idxB.zero.hr,:),BAL0tot(i,idxB.zero.min,:),BAL0tot(i,idxB.zero.sec,:)],[3 2 1]);
% 
%             % define year/month/day -> measurement data are always taken on the 
%             % same calendar day (so y/m/d is set to 0,0,0), except when 
%             % datapoints are taken after midnight, then y/m/d = 0,0,1
%             yrMonDay0 = zeros(size(hrMinSec0)); % set all to 0,0,0
%             yrMonDay0(hrMinSec0(:,1)<6,3) = 1; % correct for datapoints taken after midnight (which is considered here as before 6 am)
%             t0 = datenum([yrMonDay0,hrMinSec0]);
% 
%             % compute interpolated zero-offset (interpolated between all
%             % available zero measurement points)
%             if tMeas(i) >= max(t0) % first check whether the measurement point was not taken AFTER the last zero-measurement
%                 tMeas(i) = max(t0); % in such case set the measurement time to the time of the last zero-measurement (to prevent extrapolation)
%             end
%             BAL.BAL0.intp(i,:) = interp1(t0,permute(BAL0tot(i,idxB.zero.B,:),[3 2 1]),tMeas(i));        
%         end
% 
%     else % only one zero measurement is available
%         BAL.BAL0.intp = BAL.BAL0.raw{1}.avg(:,idxB.zero.B);
%     end
% 
%     % subtract zero measurement data from the measured data
%     BAL.B16zeroed = BAL.raw(:,idxB.raw.B)-BAL.BAL0.intp;
% %     BAL0subt = BAL.BAL0.intp - BAL.BAL0.intp(1,:);
% %     BAL.B16zeroed = BAL.raw(:,idxB.raw.B)-BAL0subt;
% 
% elseif strcmpi(zeroMode,'intp')
%     error('This zero mode has not been implemented yet -> use the sweep approach.')
%     
% 
% %     % get the angles of attack in the zero-measurement data 
% %     for i=1:length(BAL0)
% %         AoA0{i} = round(BAL0{i}.avg(:,idxB.zero.AoA)*20)/20; % unique angle(s) of attack of zero measurements     
% %     end    
% %     
% %     % get the angles of attack of the measurement data
% %     AoAmeas = round(BAL.raw(:,idxB.raw.AoA)*20)/20; % unique angle(s) of attack of measurements
% %     
% %     % loop over the angles of attack of the measurement data and get
% %     % interpolated zero offset for each angle of attack
% %     for i=1:length(AoAmeas)
% %         
% %         % get the time of the current measurement
% %         hrMinSec = [BAL.raw(:,idxB.raw.hr,:),BAL.raw(:,idxB.raw.min,:),BAL.raw(:,idxB.raw.sec,:)];
% %         yrMonDay = zeros(size(hrMinSec)); % set all to 0,0,0
% %         yrMonDay(hrMinSec(:,1)<6,3) = 1;
% %         tMeas = datenum([yrMonDay,hrMinSec]);  
% % 
% %         % loop over the zero measurement data -> if there is only 1 then no
% %         % interpolation is performed/desirable
% %         if length(AoA0) > 1 % if there is more than 1 zero measurement
% %             t0 = [];
% %             b0 = [];
% %             for j=1:length(AoA0)
% % 
% %                 %find index of angle of attack in the zero measurement data 
% %                 idx0 = find(ismember(AoA0{j},AoAmeas(i)));
% % 
% %                 % get times of these zero measurements
% %                 hrMinSec0{j} = [BAL0{j}.raw(idx0,idxB.zero.hr),BAL0{j}.raw(idx0,idxB.zero.min),BAL0{j}.raw(idx0,idxB.zero.sec)];
% %                 yrMonDay0{j} = zeros(size(hrMinSec0{j})); % set all to 0,0,0
% %                 yrMonDay0{j}(hrMinSec0{j}(:,1)<6,3) = 1; % correct for datapoints taken after midnight (which is considered here as before 6 am)
% %                 t0 = [t0;datenum([yrMonDay0{j},hrMinSec0{j}])];
% %                 b0 = [b0;BAL0{j}.raw(idx0,idxB.zero.B)];
% % 
% %                 % perform interpolation once we have reached the last zero
% %                 % measurement
% %                 if j == length(AoA0)
% %                     bal0_intp(i,:) = interp1(t0,b0,tMeas(i),'linear','extrap');
% %                 end
% % 
% %                 % if you're getting an error here, you're probably using an old version
% %                 % of MATLAB, of which the interp1 function doesn't support interpolation
% %                 % with a matrix as input. In that case use a loop approach for each of
% %                 % the balance channels:
% %                     %     for j=1:6
% %                     %         bal0_intp(:,j) = interp1(AoAzero_uniq,bal0_avg(:,idxB.zero.B(j)),AoAmeas,'pchip');
% %                     %     end
% % 
% %             end % end for loop over zero measurements
% %         
% %         else % there is only one zero measurement
% %             [AoA0srt,idxAoA0] = sort(AoA0{i}); 
% %             interp1(AoA0srt,BAL0{1}.raw(idxAoA0,idxB.zero.B),AoAmeas,'linear')
% %             
% %         end
% %         
% %         
% %     end % end for loop over angles of attack 
% %     
% %     % subtract zero measurement data from the measured data
% %     BAL.B16zeroed = BAL.raw(:,idxB.raw.B)-bal0_intp;
% 
% else
%     error('Invalid input for variable zeroMode. Change in input file main.m.')
% end % end if statement zeroMode
% end % end of function BAL_zero