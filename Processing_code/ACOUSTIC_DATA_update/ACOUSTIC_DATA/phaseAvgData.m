% %% Function phaseAvg.m
% Performs averaging of measurement data over rotation using 1P trigger 
% =========================================================================
% Tomas Sinnige - T.Sinnige@tudelft.nl 
% TU Delft - LR - AWEP - Flight Performance and Propulsion
%
% Version: 3.0
% Last updated:  28 March 2018
% First version: 02 October 2015
% =========================================================================
% | Version |    Date   |   Author  |              Changelog              |
% |---------|-----------|-----------|-------------------------------------|
% |   3.0   | 28/03/'18 | T.Sinnige | Removed unnecessary code            |
% |---------|-----------|-----------|-------------------------------------|
% |   2.0   | 11/10/'15 | T.Sinnige | -) cleaned code                     |
% |         |           |           | -) updated comments                 |
% |---------|-----------|-----------|-------------------------------------|
% |   1.0   | 02/10/'15 | T.Sinnige | First version                       |
% |---------|-----------|-----------|-------------------------------------|
% =========================================================================
% Inputs: y        - measurement data (matrix or vector)
%         oneP     - 1P data (vector)
%         fS       - sampling rate [Hz]
%         RPS      - rotational speed [1/s]
%         NrevOut  - number of output revolutions [-]
%         phIntp   - circumferential-coordinate grid [rad]
%		  dPh      - phase offset between input measurement data (y) and
% 					 measured 1P signal (oneP) - this is only nonzero in 
% 			         case there is some delay in the DAQ system)
% -------------------------------------------------------------------------  
% Output: yAvg     - phase-averaged output data
%         yStd     - standard deviation of data per phase angle
%         yRevIntp - interpolated data for each individual revolution
%         fSintp   - sampling rate interpolated data [Hz]
%         RPS1P    - time history of propeller RPS from 1P data [Hz]
% =========================================================================
function [yAvg,yStd,yRevIntp,fSintp,RPS1P,yRevStore] = phaseAvgData(y,oneP,fS,RPS,NrevOut,phIntp,dPh)

%% display progress update
% disp('Performing phase-averaging')

%% get measurement time vector
nS    = length(oneP);
tMeas = nS/fS; % total measurement time [s]
t	  = 0 : tMeas/(nS-1) : tMeas; % generate time vector [s]

%% now find the indices of the rising and falling parts of the 1P triggers
minPeakHeight = -0.0001; % define voltage value which is considered as cut-off for definition of rising and falling edges of the 1P signal
% tDelMin       = round((1/RPS)*fS*0.9); % use estimate of minimum number of samples between 1P peaks (0.9*computed number of samples based on mean RPS)
% [~,locsPos]   = findpeaks(oneP,'MinPeakHeight',minPeakHeight,'MinPeakDistance',tDelMin); % locations of positive peaks in fx --> indicating where the 1P signal is rising
% [~,locsNeg]   = findpeaks(-oneP,'MinPeakHeight',minPeakHeight,'MinPeakDistance',tDelMin); % locations of negative peaks in fx --> indicating where the 1P signal is falling
[~,locsPos]   = findpeaks(oneP,'MinPeakHeight',minPeakHeight); % locations of positive peaks in fx --> indicating where the 1P signal is rising
[~,locsNeg]   = findpeaks(-oneP,'MinPeakHeight',minPeakHeight); % locations of negative peaks in fx --> indicating where the 1P signal is falling

%% now find the indices of the mid points of the peaks in the 1P signal
% tRPS = 1/RPS; 

if ~isempty(locsPos) && ~isempty(locsNeg)
    % remove indices of any downgoing peaks that occur before the first upgoing
    locsNeg(locsNeg<locsPos(1)) = [];
    
    % remove indices of any upgoing peaks that occur after the last downgoing
    locsPos(locsPos>locsNeg(end)) = [];
    
    % OLD: take mean of rising and falling 1P indices to get center of 1P peak, this is taken as reference position
    % idx1P = round((locsPos+locsNeg)/2); 
    % NEW: take rising 1P indices to define reference position (start of 1P signal)
    idx1P = locsPos; 
    
    % % % plot 1P signal with markers related to reference point search process
    % figure('Name','1P signal')
    % hold on
    % plot(t,oneP)%,'-*','MarkerSize',4)
    % plot(t(locsPos),oneP(locsPos),'og')
    % plot(t(locsNeg),oneP(locsNeg),'sr')
    % plot(t(idx1P),oneP(idx1P),'*c')
    % hold off
    % xlim([0 0.5])
    % drawnow

    % get times at start of rotation
    t1P   = t(idx1P); % time at start of 1P trigger [s] (from 1P signal)
    RPS1P = 1./diff(t1P); % propeller rotational speed per revolution [Hz] (from 1P signal)

    if max(RPS1P)>1.1*RPS || min(RPS1P)<0.9*RPS
        disp('Warning: RPS computed from 1P trigger signal differs more than 10% from motor reading.')
        yAvg     = zeros(length(phIntp),6);
        yStd     = zeros(length(phIntp),6);
        yRevIntp = zeros(length(phIntp),6);
        fSintp   = 0;
        return
    end

else % no useable 1P signal available 
    disp('Warning: 1P trigger signal does not include any rising edges.')
    yAvg     = zeros(length(phIntp),6);
    yStd     = zeros(length(phIntp),6);
    yRevIntp = zeros(length(phIntp),6);
    RPS1P    = 0;
    fSintp   = 0;
    return
end

RPS1Pavg = mean(RPS1P); % propeller rotational speed [Hz] (from 1P signal)

%% select data from entire useable time history
lRev    = diff(idx1P);         % length of data corresponding to the individual revolutions for which microphone data is available
Nrev    = length(lRev);        % total number of full revolutions

%% Divide data into revolutions and interpolate data towards constant polar angle grid within revolution
% Nsens    = size(y,2); % number of sensors 

% get offset between 1P signal and signal(s) of interest
dIdxSig1P = round((dPh/360)*(1/RPS)*fS);
idx1P = idx1P-dIdxSig1P;

Nrev = Nrev-1; % skip last revolution to avoid problems with telemetry system delay
if idx1P(1)<0
    idx1P(1) = [];
    Nrev = Nrev-1;
end

NavgSeg  = floor(Nrev/NrevOut); % total number of averaging segments
NrevAvg  = floor(Nrev/NavgSeg); % total number of full revolutions after averaging
yRevIntp = zeros(length(phIntp),size(y,2),Nrev); % initialize output array


for j=1:Nrev % loop over all revolutions 
    
    % get indices corresponding to current rotation (plus 1st point of
    % next rotation)
    % idxRev = (idx1P(j):idx1P(j+1)-1)-dIdxSig1P;
    idxRev = (idx1P(j):idx1P(j+1));

    % get time vector corresponding to current rotation (plus 1st point of
    % next rotation)
    tRev   = t(idxRev); % [s]
    phRev  = (tRev-tRev(1))/(tRev(end)-tRev(1))*2*pi; % [rad]
    
    % get data corresponding to current rotation (plus 1st point of
    % next rotation)
    yRev = y(idxRev,:);
    yRevStore{j}=yRev;
    yRevStore{j}(:,end+1)=oneP(idxRev);

    % interpolate towards constant theta grid
    yRevIntp(:,:,j) = interp1(phRev,yRev,phIntp,'linear'); % interpolated
    
    % if you're getting an error here, you're probably using an old version 
    % of MATLAB, of which the interp1 function doesn't support interpolation
    % with a matrix as input. In that case use the following loop approach
    % and comment out the permute statement underneath the for loop. 
%     for k=1:Nsens % loop over all sensors
%         yRevIntp(:,j,k) = interp1(phRev,yRev,phIntp,'linear'); % interpolated
%     end % end for loop over all microphones
end % end for loop over all revolutions
yRevIntp = permute(yRevIntp,[1 3 2]);

% figure
% plot(thIntp,mean(plot(thIntp,yRevIntp(:,i,2)),2))
yRevIntpSeg_cell = arrayfun(@(i) yRevIntp(:,1+(i-1)*NrevAvg:i*NrevAvg,:),1:NavgSeg,'UniformOutput',0);
yRevIntpSeg      = cat(4,yRevIntpSeg_cell{:});
yAvgPerRev       = mean(yRevIntpSeg,4);
yStdPerRev       = std(yRevIntpSeg,[],4);

%% compute final sampling rate
tRPS     = 1/RPS;                       % (average) time per revolution [s]
tIntpRev = tRPS*(phIntp/(2*pi));        % time vector for average rotation [s]
fSintp   = 1/(tIntpRev(2)-tIntpRev(1)); % sampling rate for average rotation [Hz]

%% reshape vector
yAvg = [];
yStd = [];
for i=1:size(yAvgPerRev,2)
    temp1 = permute(yAvgPerRev(:,i,:),[1 3 2]);
    temp2 = permute(yStdPerRev(:,i,:),[1 3 2]);
    yAvg = [yAvg;temp1];
    yStd = [yStd;temp2];
end

% figure,
% for j=1:6
%     subplot(3,2,j),hold on
%     % for i=1:length(yRev)
%     %     plot(phRev{i},yRev{i}(:,j),'*')
%     % end
%     plot(phIntp,yAvgPerRev(:,:,j),'-r')
%     plot(phIntp,yAvg(:,j),'--k')
% end
% drawnow 

end % end of function phaseAvg_magnetic.m