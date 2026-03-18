close all
clear
clc

%% Inputs
% path to folder containing the measurement data
fnFolder = '.\DATA'

% structure of filenames to main txt file containing the averaged data - you can add multiple filenames here
fn = {'propOn_dE000_dR000_J16.txt'}; 

% propeller diameter (used to compute advance ratio)
D = 0.4064;

% inputs for phase averaging
phIntp = linspace(0,2*pi,361); % azimuthal grid for phase averaging 
phIntp(end)=[]; % remove the grid point at 2*pi - phase-averaged signal is periodic so should be the same as at 0 deg
dPh = 0; % offset in phase between measurement data and 1P signal data
fS = 51.2e3; % sampling frequency [Hz]

%% Loop over all TDMS files of name "Measurement_i.tdms)" in the specified folder
for i=1:length(fn)
   
    % load data operating file
    AVGpath    = [fnFolder,'\',fn{i}];
    AVGdata{i} = load(AVGpath);
    
    opp{i}.DPN    = AVGdata{i}(:,1);
    opp{i}.vInf   = AVGdata{i}(:,7); % freestream velocity [m/s]
    opp{i}.AoA    = AVGdata{i}(:,13);  % angle of attack [deg]
    opp{i}.AoS    = AVGdata{i}(:,14);  % angle of sideslip [deg]
    opp{i}.RPS_M1 = AVGdata{i}(:,15);  % RPS motor 1 [Hz]
    opp{i}.RPS_M2 = AVGdata{i}(:,22);  % RPS motor 2 [Hz]
    opp{i}.J_M1   = opp{i}.vInf./(opp{i}.RPS_M1*D); % advance ratio motor 1
    opp{i}.J_M2   = opp{i}.vInf./(opp{i}.RPS_M2*D); % advance ratio motor 2
    
    % load microphone data
    for j=1:length(opp{i}.DPN) % loop over all the datapoints for this configuration
        
        % Construct filename (required in case of duplicate files)
        runNo = 1;
        TDMSpath = [fnFolder '\' fn{i}(1:end-4) '_run',num2str(opp{i}.DPN(j)),'_',sprintf('%03.0f',runNo),'.tdms'];
        
        % load data
        rawData = ReadFile_TDMS(TDMSpath);
        disp(['Loaded file ' TDMSpath])
        
        % Extract the microphone pressure and write to 
        MIC{i}.pMic{j} = rawData{1}(:,1); % the data are stored with calibration factor applied [Pa]
        MIC{i}.oneP{j} = rawData{1}(:,2:3); % these are the data from the one-per-revolution trigger (pulse whenever the propellers pass a predefined azimuthal position)
    
        % Perform phase-averaging of signals (optional)
        [MIC{i}.yAvg(:,j),~,~,~,~,~] = phaseAvgData(MIC{i}.pMic{j},MIC{i}.oneP{j}(:,1),fS,opp{i}.RPS_M1(j),1,phIntp,dPh);

    end
    
end % end while loop over files


%% Plot phase-averaged data
figure
for j=1:2
subplot(2,1,j)
plot(phIntp,MIC{1}.yAvg(:,j))
title(['AoA=',sprintf('%.1f',opp{1}.AoA(j)),'deg'])
xlabel('Phase angle [rad]')
ylabel('Acoustic pressure [Pa]')
end


figure,plot((opp{i}.AoA),rms(MIC{i}.yAvg),'*b')
xlabel('AoA [deg]')
ylabel('pRMS tonal content [Pa]')
