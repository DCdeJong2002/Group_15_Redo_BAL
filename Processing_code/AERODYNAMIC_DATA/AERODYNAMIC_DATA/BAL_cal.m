%% Function BAL_cal.m
% Function used to calibrate LTT balance data
% =========================================================================
% Tomas Sinnige - T.Sinnige@tudelft.nl 
% TU Delft - LR - AWEP - Flight Performance and Propulsion
%
% Version: 1.2
% Last updated:  26 February 2021
% First version: 07 October 2017
% =========================================================================
% | Version |    Date   |   Author  |              Changelog              |
% |---------|-----------|-----------|-------------------------------------|
% |   1.2   | 26/02/'21 | T.Sinnige | Changed variable name non-linear    |
% |         |           |           | caliration coefficients             |
% |---------|-----------|-----------|-------------------------------------|
% |   1.1   | 17/10/'17 | T.Sinnige | Moved load operating stiffness      |
% |         |           |           | matrix to function BAL_getCalFactors|
% |---------|-----------|-----------|-------------------------------------|
% |   1.0   | 16/10/'17 | T.Sinnige | Vectorized code                     |
% |---------|-----------|-----------|-------------------------------------|
% |   0.0   | 07/10/'17 | T.Sinnige | First version                       |
% |---------|-----------|-----------|-------------------------------------|
% =========================================================================
% Inputs:  bal    - vector containing the raw balance data (steps)
%          p      - linear calibration matrix LTT balance
%          pnl      - nonlinear calibration matrix LTT balance
%          arm    - vector containing moment arms 
%          FX_cor - drag force correction parameter to account for 
%                   influence of FZ
%          x_bend - bending characteristics x-direction
%          y_bend - bending characteristics y-direction
%          e      - stiffness matrix LTT balance
% -------------------------------------------------------------------------
% Outputs: F      - force vector [N] ([FX,FY,FZ])
%          M      - moment vector[Nm] ([MX,MY,MZ])
% =========================================================================
function [F,M] = BAL_cal(bal,p,pnl,arm,FX_cor,x_bend,y_bend,e)

%% Compute force components
f = pnl*(p.*bal).'; % calibrated balance data

%% Compute forces
F(1) = f(1);
F(2) = f(2)+f(6);
F(3) = f(3)+f(4)+f(5);

% correct FX for influence of FZ
F(1) = F(1) + FX_cor*F(3);

%% Compute moments
% compute the new moment arms under loading (due to deformation of balance)
arm_new = arm + (e*f).';

% compute moments
M(1) = -f(2)*arm_new(11)-f(6)*arm_new(12)+f(3)*arm_new(7)+f(4)*arm_new(8)+f(5)*arm_new(9);
M(2) = +f(1)*arm_new(10)-f(3)*arm_new(2) -f(4)*arm_new(3)-f(5)*arm_new(4);
M(3) = -f(1)*arm_new(6) +f(2)*arm_new(1) +f(6)*arm_new(5);

% correct moments for bending effects
M(1) = M(1) - F(3)*F(2)*y_bend;
M(2) = M(2) + F(3)*F(1)*x_bend;
M(3) = M(3) + F(2)*F(1)*y_bend - F(1)*F(2)*x_bend;

end % end of function BAL_cal.m