%==========================================================================
% Contact: Woutijn J. Baars, w.j.baars@tudelft.nl
%
% Input:   e.g. time-series of acoustic pressure data
%             - units: [Pa]
%
% Objective: generates 1D spectrum
%
%==========================================================================


%% SCRIPT------------------------------------------------------------------
clear all; clc; close all;


%% SIGNAL------------------------------------------------------------------

%%%% INPUT: LOAD DATA, e.g., TIME SERIES OF PRESSURE %%%%%%%%%%%%%%%%%%%%%%
%%%%
load data_example_jetnoise; % example off a jet noise experiment
%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fs = 51200; % sampling frequency, [Hz]

% plot time-series of microphone signal
figure;
Naq = length(data);    % # of acquisition samples
dt = 1/fs;              % sample time, [s]
plot(0:dt:dt*(Naq-1),data);
axis([0 65 -20 20]); xlabel('t (s)'); ylabel('p (Pa)');
title('Time series of acoustic pressure in Pa');


%% SPECTRAL ANALYSIS-RAW---------------------------------------------------

%%%% INPUT: SIZE OF FFT-BLOCK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%
N = 2^13;           % ensemble size for Fourier analysis, [-]
                    % sets lowest resolved frequency
                    % THIS 'N' you can adjust, e.g., N = 2^11, or N = 2^16
                    % When this number is larger: better spectral
                    % resolution, but less 'ensemble averaging'...
%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

%%%% INPUT: REFERENCE PRESSURE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%
p_ref = 20e-6;      % 20 micro-Pa reference pressure, [Pa]
%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

B = floor(Naq/N);   % # of enemble averages per data file (Nfile samples), [-]
df = fs/N;          % frequency resolution, [Hz]
flab = (0:N-1)*df;  % frequency discretization, [Hz]
T = dt*N;           % recording time one partition, [s]
windowing = true;   % hannding window or not? (not much different for broadband signal)

% OASPL [dB] – SINGLE NUMBER FOR A TIME SERIES
ApOASPL_dB = 20*log10(std(data)/p_ref);

% function for spectrum:
[~,flab,~,df,phi,N] = fcn_spectrumN_V1(N,1/fs,data,2); % [bst = 1: spatial-data, bst = 2: time-data]

% Parseval's theorem: check scaling-----------------------------------
% signal's energy - integrating the spectrum:
p2int = trapz(phi(1:N/2,1))*df; % Trapezoidal-integration: Int[PSD function(f)]df, [unit^2 = energy], slightly less accurate: p2int = sum(Guu(1:N/2,1))*df;
% signal's energy - same as variance? ratio = 1?
display(strcat(['Ratio = ',num2str(p2int/(std(data)).^2)])); % CHECK?
% --------------------------------------------------------------------

% SPSL: [dB/Hz] - SPECTRUM IN DB/Hz
SPSL = 20*log10(sqrt(phi/p_ref^2));

% spectrum in dB/Hz (so basically log-log)
figure;
semilogx(flab(1:N/2),SPSL(1:N/2));
axis([1 100000 -10 100]); xlabel('f (Hz)'); ylabel('SPSL (dB/Hz)');
title('Acoustic spectrum in dB/Hz, with log-axis of frequency');


%% END