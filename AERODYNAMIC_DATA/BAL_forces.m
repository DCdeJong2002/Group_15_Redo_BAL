%% Function BAL_forces.m
% Computes balance forces (LTT)
% =========================================================================
% Tomas Sinnige - T.Sinnige@tudelft.nl 
% TU Delft - LR - AWEP - Flight Performance and Propulsion
%
% Version: 1.2
% Last updated:  26 February 2021
% First version: 12 October 2017
% =========================================================================
% | Version |    Date   |   Author  |              Changelog              |
% |---------|-----------|-----------|-------------------------------------|
% |   1.2   | 26/02/'21 | T.Sinnige | -) Removed PRS references           |
% |---------|-----------|-----------|-------------------------------------|
% |   1.1   | 17/10/'17 | T.Sinnige | -) Added computation advance ratio  |
% |         |           |           | -) Commented code 'intp' zeroMode   |
% |         |           |           |     -> the sweep option works fine  |
% |---------|-----------|-----------|-------------------------------------|
% |   1.0   | 16/10/'17 | T.Sinnige | -) Added interpolation (in time)    |
% |         |           |           |    between multiple zero-m'ments    |
% |---------|-----------|-----------|-------------------------------------|
% |   0.0   | 12/10/'17 | T.Sinnige | First version                       |
% |---------|-----------|-----------|-------------------------------------|
% =========================================================================
% Inputs:  BAL     - structure containing measurement data balance
%          BAL0    - structure containing zero-measurement data balance
%          idxB    - structure containing indices balance data (W3D)
%          D        - propeller diameter [m]
%          S        - reference area [m^2]
%          b        - wing span [m]
%          c        - mean aerodynamic chord [m]
%          XmRefB   - moment reference points (x,y,z coordinates) in
%                     balance reference system [1/c] 
%          XmRefM   - moment reference points (x,y,z coordinates) in model 
%                     reference system [1/c] 
%          dAoA     - angle of attack offset (subtracted from measured 
%                     values)   
%          dAoS     - angle of sideslip offset (subtracted from measured 
%                     values)
%          modelType- type of wind-tunnel model
%          modelPos - orientation of wind-tunnel model
%          testSec - LTT test section number
% -------------------------------------------------------------------------
% Outputs: BAL     - structure containing measurement data balance
%                    (with processed forces and moments added)
%                      BAL.raw       -> raw data (from raw_.. file)
%                      BAL.BAL0      -> zero-measurement data
%                      BAL.B16zeroed -> raw data of B1-6 (steps), with zero
%                                       offset removed
% =========================================================================
function [BAL] = BAL_forces(BAL,BAL0,idxB,D,S,b,c,XmRefB,XmRefM,dAoA,dAoS,modelType,modelPos,testSec)

%% Compute operating conditions
% get operating conditions 
oper = SUP_LTTgetOper_BAL(BAL,testSec);

% write updated data to BAL variable
BAL.q     = oper.qInf; % freestream dynamic pressure [Pa] 
BAL.rho   = oper.rho;  % freestream air density [kg/m^3]
BAL.V     = oper.vInf; % freestream velocity [m/s]
BAL.temp  = oper.tInf; % freestream temperature [K]
BAL.pInf  = oper.pInf; % freestream static pressure [Pa]
BAL.pBar  = oper.pBar; % barometric pressure [Pa]
BAL.nu    = oper.nu;   % freestream kinematic viscosity [m^2/s]

%% Compute advance ratio
BAL.J_M1 = BAL.V./(BAL.rpsM1*D);
BAL.J_M2 = BAL.V./(BAL.rpsM2*D);

%% Compute Reynolds number 
BAL.Re = BAL.V*c./BAL.nu;

%% Get balance calibration factors
[p,pnl,arm,FX_cor,x_bend,y_bend,e] = BAL_getCalFactors;

%% Remove zero offset from balance data
BAL = BAL_zero(BAL,BAL0,idxB);

%% Perform balance calibration
for i=1:size(BAL.B16zeroed,1)
    [F(i,:),M(i,:)] = BAL_cal(BAL.B16zeroed(i,:),p,pnl,arm,FX_cor,x_bend,y_bend,e);
end

%% Compute nondimensional forces and moments
CF(:,1) = F(:,1) ./ (oper.qInf*S); % force in x-direction balance reference system
CF(:,2) = F(:,2) ./ (oper.qInf*S); % force in y-direction balance reference system
CF(:,3) = F(:,3) ./ (oper.qInf*S); % force in z-direction balance reference system
CM(:,1) = M(:,1) ./ (oper.qInf*S*b); % moment about x-axis balance reference system
CM(:,2) = M(:,2) ./ (oper.qInf*S*c); % moment about y-axis balance reference system
CM(:,3) = M(:,3) ./ (oper.qInf*S*b); % moment about z-axis balance reference system

%% Redefinition of the aerodynamic coefficients in model reference system
% get angles of incidence
AoA = BAL.AoA - dAoA; % correct AoA for constant AoA offset [deg]
AoS = BAL.AoS - dAoS; % correct AoS for constant AoS offset [deg]
BAL.AoA = AoA;
BAL.AoS = AoS;

% compute forces and moments 
if strcmpi(modelType,'aircraft') || strcmpi(modelType,'3dwing')
    % forces in model axis system
    CFt = CF(:,1).*cosd(AoA) - CF(:,3).*sind(AoA); % tangential force [N]
    CFn = CF(:,3).*cosd(AoA) + CF(:,1).*sind(AoA); % normal force [N]
    CFs = -CF(:,2); % side force [N]
    
    % moments with respect to the moment point in the balance axis system (XmRef)
    CMr = CM(:,1) - XmRefB(2)*CF(:,3) + XmRefB(3)*CF(:,2); % rolling moment [Nm]
    CMp = CM(:,2) - XmRefB(3)*CF(:,1) + XmRefB(1)*CF(:,3); % pitching moment [Nm]
    CMy = CM(:,3) + XmRefB(2)*CF(:,1) - XmRefB(1)*CF(:,2); % yawing moment [Nm] 

    % recalculate moments in the airplane axis system
    CMr = CMr.*cosd(AoA) - CMy.*sind(AoA);
    CMp = CMp;
    CMy = CMy.*cosd(AoA) + CMr.*sind(AoA);
    
    % account for model orientation (upper surface pointing down or up in
    % the wind tunnel)
    if strcmpi(modelPos,'normal')
        CFt = +CF(:,1).*cosd(AoA) + CF(:,3).*sind(AoA); % tangential force [N]
        CFn = -CF(:,3).*cosd(AoA) + CF(:,1).*sind(AoA); % normal force [N]
        CMr = -CMr;
        CMp = -CMp;
        CMy = -CMy;
    end
    
elseif strcmpi(modelType,'halfwing') % for this case, only the case of angle-of-attack variations is programmed (sideslip always zero)

    % forces
    CFt = CF(:,1); % tangential force [N]
    CFn = CF(:,2); % normal force [N]
    CFs = CF(:,3); % side force [N]
    
    % moments
    CMr = CM(:,1)        + XmRefB(3)*CF(:,2)*(c/b) - XmRefB(2)*CF(:,3)*(c/b); % rolling moment [Nm]
    CMp = -CM(:,3)*(c/b) - XmRefB(2)*CF(:,1)       + XmRefB(1)*CF(:,2);       % pitching moment [Nm]
    CMy = CM(:,2)*(c/b)  + XmRefB(1)*CF(:,3)*(c/b) - XmRefB(3)*CF(:,1)*(c/b); % yawing moment [Nm] 

    % account for model orientation (upper surface pointing down or up in
    % the wind tunnel)
    if strcmpi(modelPos,'inverted')
        CFn = -CFn; % normal force [N]
        CMr = -CMr;
        CMp = -CMp;
        CMy = -CMy;
    end
   
end % end if statement model orientation

%% Compute lift/drag/pitching moment
CFl   =  (CFn.*cosd(AoA)-CFt.*sind(AoA)); % lift
CFd   =  (CFn.*sind(AoA)+CFt.*cosd(AoA)).*cosd(AoS)+CFs.*sind(AoS); % drag
CFyaw = -(CFn.*sind(AoA)+CFt.*cosd(AoA)).*sind(AoS)+CFs.*cosd(AoS); % sideforce
CMp25c = CMp + CFn*(0.25-XmRefM(1))-CFt*(0.0-XmRefM(3));
    
%% Write forces to BAL data structure
BAL.FX  = F(:,1);
BAL.FY  = F(:,2);
BAL.FZ  = F(:,3);
BAL.MX  = M(:,1);
BAL.MY  = M(:,2);
BAL.MZ  = M(:,3);

BAL.CFX = CF(:,1);
BAL.CFY = CF(:,2);
BAL.CFZ = CF(:,3);
BAL.CMX = CM(:,1);
BAL.CMY = CM(:,2);
BAL.CMZ = CM(:,3);

BAL.CN  = CFn;
BAL.CT  = CFt;
BAL.CY  = CFs;

BAL.CL  = CFl;
BAL.CD  = CFd;
BAL.CYaw  = CFyaw;

BAL.CMroll      = CMr;
BAL.CMpitch     = CMp;
BAL.CMpitch25c  = CMp25c;
BAL.CMyaw       = CMy;

BAL.b  = b;
BAL.c  = c;
BAL.S  = S;

end % end of function BAL_forces