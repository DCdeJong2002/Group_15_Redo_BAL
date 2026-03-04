function [oper] = SUP_LTTgetOper_BAL(DAT,testSec)

    % get data from PRS data structure
    oper.dPb  = DAT.dPb;         % tunnel dynamic pressure
    oper.tInf = DAT.temp+273.15; % tunnel temperature [K]
    oper.pBar = DAT.pBar*100;    % barometric pressure [Pa]
    oper.AoA  = DAT.AoA;         % angle of attack [deg]
    
    % compute dynamic pressure from tunnel calibration
    oper.qInf = SUP_LTTq(oper.dPb,testSec); % dynamic pressure [Pa]
        
    % assume static pressure in windtunnel is the same as the barometric
    % pressure (cannot be done without pressure-probe data)
    oper.pInf = oper.pBar; % static pressure in windtunnel [Pa]
    
    % compute density 
    oper.rho  = oper.pInf./(oper.tInf*287.05); % air density [kg/m3]

    % compute freestream velocity
    oper.vInf = sqrt(2*oper.qInf./oper.rho);
    
    % compute viscosity
    mu = 1.716e-5*(oper.tInf/273.15).^1.5.*(273.15+110.4)./(oper.tInf+110.4); % dynamic viscosity from Sutherland's law
    oper.nu = mu./oper.rho; % kinematic viscosity [m^2/s]
    
end % end of function SUP_LLTgetOper