%% Function SUP_getIdx.m
% Defines indices of LTT data files.
% =========================================================================
% Tomas Sinnige - T.Sinnige@tudelft.nl 
% TU Delft - LR - AWEP - Flight Performance and Propulsion
%
% Version: 1.1
% Last updated:  17 October 2017
% First version: 10 October 2017
% =========================================================================
% | Version |    Date   |   Author  |              Changelog              |
% |---------|-----------|-----------|-------------------------------------|
% |   1.1   | 16/10/'17 | T.Sinnige | Added indices raw balance data for  |
% |         |           |           | calibrated moments, coefficients    |
% |---------|-----------|-----------|-------------------------------------|
% |   1.0   | 16/10/'17 | T.Sinnige | Added indices raw balance data for  |
% |         |           |           | calibrated forces, dynamic pres.    |
% |---------|-----------|-----------|-------------------------------------|
% |   0.0   | 10/10/'17 | T.Sinnige | First version (from previous tests) |
% |---------|-----------|-----------|-------------------------------------|
% =========================================================================
% Inputs:  none
% -------------------------------------------------------------------------
% Outputs: idxB - structure containing indices balance data (W3D)
%                   idxB.raw  -> raw data file
%                   idxB.unc  -> uncorrected data file
%                   idxB.cor  -> corrected data file
%                   idxB.zero -> zero-measurement data file
%          idxP - structure containing indices pressure data (ProfMeasure)
%                   idxP.raw  -> raw data file
%                   idxP.unc  -> uncorrected data file
%                   idxP.cor  -> corrected data file
% =========================================================================
function [idxB,idxP] = SUP_getIdx


%% Define indices BALANCE DATA
% raw data file
idxB.run   = 1;     % run number
idxB.hr    = 2;     % time of measurement: hour
idxB.min   = 3;     % time of measurement: min
idxB.sec   = 4;     % time of measurement: seconds
idxB.AoA   = 5;     % angle of attack [deg]
idxB.AoS   = 6;     % angle of sideslip [deg]
idxB.dPb   = 7;     % pressure delta measured upstream in tunnel [Pa]
idxB.pBar  = 8;     % barometric pressure [Pa]
idxB.temp  = 9;     % freestream temperature [K]
idxB.B     = 10:15; % raw balance data [steps]
idxB.B1    = 10;    % raw balance data, component B1 [steps]
idxB.B2    = 11;    % raw balance data, component B2 [steps]
idxB.B3    = 12;    % raw balance data, component B3 [steps]
idxB.B4    = 13;    % raw balance data, component B4 [steps]
idxB.B5    = 14;    % raw balance data, component B5 [steps]
idxB.B6    = 15;    % raw balance data, component B6 [steps]
idxB.rpmWT = 16;    % wind tunnel RPM [1/min]
idxB.rho   = 17;    % freestream density [kg/m^3]
idxB.q     = 18;    % freestream dynamic pressure [Pa]
idxB.V     = 19;    % freestream velocity [m/s]
idxB.Re    = 20;    % Reynolds number
idxB.rpsM1 = 21;    % RPS of motor #1 [Hz] (not used for PROWIM)
idxB.rpsM2 = 22;    % RPS of motor #2 [Hz] (used for PROWIM)
idxB.iM1   = 23;    % current of motor #1 [A] 
idxB.iM2   = 24;    % current of motor #2 [A] 

idxB.dPtQ  = 25;    % ?

idxB.tM1   = 26;    % temperature of motor #1 [deg C] 
idxB.tM2   = 27;    % temperature of motor #2 [deg C] 
idxB.vM1   = 28;    % voltage of motor #1 [V] 
idxB.vM2   = 29;    % voltage of motor #2 [V] 



% idxB.FX    = 26;    % force in X-direction [N]
% idxB.FY    = 27;    % force in Y-direction [N]
% idxB.FZ    = 28;    % force in Z-direction [N]
% idxB.MX    = 29;    % moment in X-direction [Nm]
% idxB.MY    = 30;    % moment in X-direction [Nm]
% idxB.MZ    = 31;    % moment in X-direction [Nm]
% 
% idxB.CFX   = 32;    % force coefficient in X-direction [N]
% idxB.CFY   = 33;    % force coefficient in Y-direction [N]
% idxB.CFZ   = 34;    % force coefficient in Z-direction [N]
% idxB.CMX   = 35;    % moment coefficient in X-direction [Nm]
% idxB.CMY   = 36;    % moment coefficient in X-direction [Nm]
% idxB.CMZ   = 37;    % moment coefficient in X-direction [Nm]
% 
% idxB.N     = 38;    % normal force [N]
% idxB.T     = 39;    % tangential force [N]
% idxB.Y     = 40;    % side force [N]
% idxB.L     = 41;    % lift [N]
% idxB.D     = 42;    % drag [N]
% idxB.Mr    = 43;    % rolling moment [Nm]
% idxB.Mp    = 44;    % pitching moment [Nm]
% idxB.My    = 45;    % yawing moment [Nm]
% 
% idxB.CN    = 46;    % normal-force coefficient
% idxB.CT    = 47;    % tangential-force coefficient
% idxB.CY    = 48;    % side-force coefficient
% idxB.CL    = 49;    % lift coefficient
% idxB.CD    = 50;    % drag coefficient
% idxB.CMr   = 51;    % rolling-moment coefficient
% idxB.CMp   = 52;    % pitching-moment coefficient
% idxB.CMy   = 53;    % yawing-moment coefficient
% 
% idxB.CTp   = 54;    % prop thrust coefficient CT = T / (rho*n^2*D^4)
% idxB.TCp   = 55;    % prop thrust coefficient TC = T / (qInf*pi*R^2)
% idxB.TCw   = 56;    % prop thrust coefficient TC = T / (qInf*S)
% 
% idxB.qInf_pt = 57; % dynamic pressure based on pitot tube in test section [Pa]
% idxB.J       = 58; % advance ratio (overwritten in case of correction of CL and CD for advance-ratio shifts)
% idxB.Jori    = 59; % advance ratio (as measured)
% % idxB.CLcor   = 60; % lift coefficient corrected for advance-ratio shift w.r.t. desired advance ratios (0.7,0.8,0.9,1.0)
% % idxB.CDcor   = 61; % drag coefficient corrected for advance-ratio shift w.r.t. desired advance ratios (0.7,0.8,0.9,1.0)
% 
% idxB.CLp     = 62; % lift coefficient associated with prop only
% idxB.CDp     = 63; % drag coefficient associated with prop only
% idxB.CLwoP   = 64; % lift coefficient associated without prop contribution
% idxB.CDwoP   = 65; % drag coefficient associated without prop contribution
% 
% idxB.CLempty = 66; % lift coefficient of empty section (rotation plate is connected to balance)
% idxB.CDempty = 67; % drag coefficient of empty section (rotation plate is connected to balance)
% 
% idxB.b = 68; % wing span [m]
% idxB.c = 69; % wing chord [m]
% idxB.S = 70; % reference area used for nondimensionalization [m^2]
% 
% idxB.nu   = 71; % kinematic viscosity [m^2/s]
% idxB.pInf = 72; % freestream static pressure [Pa]
% idxB.CLori = 73;
% idxB.CDori = 74;
% 
% idxB.dPtM    = 75;
% idxB.dPsInf  = 76;
% 
% idxB.CYaw    = 77; % yawing force coefficient
% idxB.CMp25c  = 78; % pitching moment coefficient around quarter chord
% 
% 
% 
% 
% % corrected data file
% % idxB.cor.run  = 1;
% % idxB.cor.AoA  = 2;
% % idxB.cor.AoS  = 3;
% % idxB.cor.CL   = 4;
% % idxB.cor.CD   = 5;
% % idxB.cor.Cyaw = 6;
% % idxB.cor.CM   = 7;
% % idxB.cor.CT   = 8;
% % idxB.cor.CN   = 9;
% % idxB.cor.CY   = 10;
% % idxB.cor.Cl   = 11;
% % idxB.cor.Cm   = 12;
% % idxB.cor.Cn   = 13;
% % idxB.cor.V    = 14;
% % idxB.cor.Re   = 15;
% % idxB.cor.M    = 16;
% % 
% % % uncorrected data file
% % idxB.unc.run  = 1;
% % idxB.unc.AoA  = 2;
% % idxB.unc.AoS  = 3;
% % idxB.unc.CL   = 4;
% % idxB.unc.CD   = 5;
% % idxB.unc.Cyaw = 6;
% % idxB.unc.CM   = 7;
% % idxB.unc.CT   = 8;
% % idxB.unc.CN   = 9;
% % idxB.unc.CY   = 10;
% % idxB.unc.Cl   = 11;
% % idxB.unc.Cm   = 12;
% % idxB.unc.Cn   = 13;
% % idxB.unc.CX   = 14;
% % idxB.unc.CY   = 15;
% % idxB.unc.CZ   = 16;
% % idxB.unc.CmX  = 17;
% % idxB.unc.CmY  = 18;
% % idxB.unc.CmZ  = 19;
% % idxB.unc.FX   = 20;
% % idxB.unc.FY   = 21;
% % idxB.unc.FZ   = 22;
% % idxB.unc.MX   = 23;
% % idxB.unc.MY   = 24;
% % idxB.unc.MZ   = 25;
% % idxB.unc.dPb  = 26;
% % idxB.unc.pBar = 27;
% % idxB.unc.Temp = 28;
% % idxB.unc.rho  = 29;
% % idxB.unc.q    = 30;
% % idxB.unc.V    = 31;
% % idxB.unc.Re   = 32;
% % idxB.unc.M    = 33;
% 
% zero data file
idxB.zero.run   = 1;
idxB.zero.hr    = 2;
idxB.zero.min   = 3;
idxB.zero.sec   = 4;
idxB.zero.AoA   = 5;
idxB.zero.AoS   = 6;
idxB.zero.pBar  = 7;
idxB.zero.temp  = 8;
idxB.zero.B     = 9:14; 
idxB.zero.B1    = 9;
idxB.zero.B2    = 10;
idxB.zero.B3    = 11;
idxB.zero.B4    = 12;
idxB.zero.B5    = 13;
idxB.zero.B6    = 14;

%% Define indices PRESSURE DATA
% raw data
idxP.run    = 1;
idxP.hr     = 2;
idxP.min    = 3;
idxP.sec    = 4;
idxP.AoA    = 5;
idxP.AoS    = 6;
idxP.dPb    = 7;
idxP.pBar   = 8;
idxP.temp   = 9;
idxP.B1     = 10;
idxP.B2     = 11;
idxP.B3     = 12;
idxP.B4     = 13;
idxP.B5     = 14;
idxP.B6     = 15;
idxP.X      = 16; % X-coordinate traversing system
idxP.Y      = 17; % Y-coordinate traversing system
idxP.Z      = 18; % Z-coordinate traversing system
idxP.rho    = 19; % air density [kg/m3]
idxP.q      = 20; % freestream dynamic pressure [Pa]
idxP.V      = 21; % freestream velocity [m/s]
idxP.Re     = 22; % Reynolds number based on input chord length [-]
idxP.FX     = 23; % force in x-direction [N] (balance data)
idxP.FY     = 24; % force in y-direction [N] (balance data)
idxP.FZ     = 25; % force in z-direction [N] (balance data)
idxP.MX     = 26; % moment in x-direction [Nm] (balance data)
idxP.MY     = 27; % moment in y-direction [Nm] (balance data)
idxP.MZ     = 28; % moment in z-direction [Nm] (balance data)
idxP.cl     = 29;
idxP.cd     = 30; 
idxP.cm     = 31;
idxP.rps_ac = 32;
idxP.rps_sp = 33;
idxP.dF_sp  = 34;
idxP.PR     = 35;
idxP.rpmWT  = 36;
idxP.extra6 = 37;
idxP.q_wr   = 38;
idxP.dF     = 39;
idxP.dX_f   = 40; % x-displacement flap [mm] (in case of fowler flap)
idxP.dY_f   = 41; % y-displacement flap [mm] (in case of fowler flap)
idxP.pres   = 42:296; % pressures Initium [Pa]

% define indices of pressure ports
idxP.presTaps = 234:239; % pressure taps
idxP.pTfs = 240; % freestream total pressure [Pa] (relative to reference in pressure scanner)
idxP.pSfs = 241; % freestream static pressure [Pa] (relative to reference in pressure scanner)


end % end of function SUP_getIdx.m