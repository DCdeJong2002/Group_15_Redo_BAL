function [klab,flab,dk,df,Gpp,N] = fcn_spectrumN_V1(N,discr,data,bst);

% [bst = 1: spatial-data, bst = 2: time-data]

% acquisition variables -----------------------------------------
windowing = true;   % fft-windowing?
% computed:
if bst == 1;        % spatial data
    dx = discr;         % discretization step [m]
    df = 0;             % N/A frequency resolution [Hz]
    dk = 2*pi/dx/N;     % wavenumber resolution [m]
    flab = 0;           % N/A frequency discretization [Hz]
    klab = (0:N-1)*dk;  % wavenumber discretization [1/m]
    T = 1/dk;           % record length one sample [m]
    B = floor(length(data)/N); % # of ensembles (no overlap)
elseif bst == 2;    % time data
    dt = discr;         % sampling time [s]
    df = (1/dt)/N;      % frequency resolution [Hz]
    dk = 0;             % N/A wavenumber resolution [m]
    flab = (0:N-1)*df;  % frequency discretization [Hz]
    klab = 0;           % N/A wavenumber discretization [1/m]
    T = 1/df;           % record time one sample [s]
    B = floor(length(data)/N); % # of ensembles (no overlap)
end
% ---------------------------------------------------------------

% subtract mean to not have energy in zero=frequency:
data = data - mean(data);

% ensemble averaging: 50% overlap, 2*B-1 partitions
Spp = zeros(N,1);
for part = 1:2*B-1; %disp(part);
    % extract data:
    dataens = data(1+(part-1)*N/2:1+(part-1)*N/2+N-1);
    % spectra:
    if windowing == true;
        wj = window(@hann,N); % window function
        dataw = wj.*dataens;
        Spp_temp = fft(dataw);                          % FFT single partition
        Spp(:,1) = Spp(:,1) + Spp_temp.*conj(Spp_temp); % adding spectra
    else
        Spp_temp = fft(dataens);                        % FFT single partition
        Spp(:,1) = Spp(:,1) + Spp_temp.*conj(Spp_temp); % adding spectra
    end
end

% divide by # of ensembles '2*B-1':
Spp(:,1) = Spp(:,1)/(2*B-1);
% PSD-FFT Matlab scaling, this correctly scales the spectra per dk, or per df:
if windowing == true;
    Spp(:,1) = 1/N/sum(wj.^2)*Spp(:,1)*T;
else
    Spp(:,1) = 1/N^2*Spp(:,1)*T;
end

% one-sided PSD: 
Gpp = Spp*2; % [unit^2/m] for space, [unit^2/Hz] for time

return