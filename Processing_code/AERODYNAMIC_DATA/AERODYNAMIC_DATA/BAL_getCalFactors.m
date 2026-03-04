%% Function BAL_getCalFactors.m
% Function used to get the LTT balance calibration factors
% =========================================================================
% Tomas Sinnige - T.Sinnige@tudelft.nl 
% TU Delft - LR - AWEP - Flight Performance and Propulsion
%
% Version: 0.2
% Last updated:  26 February 2021
% First version: 16 October 2017
% =========================================================================
% | Version |    Date   |   Author  |              Changelog              |
% |---------|-----------|-----------|-------------------------------------|
% |   0.2   | 26/02/'21 | T.Sinnige | Changed variable name non-linear    |
% |         |           |           | caliration coefficients             |
% |---------|-----------|-----------|-------------------------------------|
% |   0.1   | 17/10/'17 | T.Sinnige | Added load operating stiffness      |
% |         |           |           | matrix from function BAL_cal        |
% |---------|-----------|-----------|-------------------------------------|
% |   0.0   | 16/10/'17 | T.Sinnige | First version                       |
% |---------|-----------|-----------|-------------------------------------|
% =========================================================================
% Inputs:  none
% -------------------------------------------------------------------------
% Outputs: p      - linear calibration matrix LTT balance
%          pnl      - nonlinear calibration matrix LTT balance
%          arm    - vector containing moment arms 
%          FX_cor - drag force correction parameter to account for 
%                   influence of FZ
%          x_bend - bending characteristics x-direction
%          y_bend - bending characteristics y-direction
%          e      - stiffness matrix balance
% =========================================================================
function [p,pnl,arm,FX_cor,x_bend,y_bend,e] = BAL_getCalFactors

% linear calibration factors balance (p1-p6)
p = [0.009200961 0.01594834 0.06134184 0.06143589 0.02461131 0.01231626]; 

% nonlinear calibration factors
pnl = [ 1            , -0.4414104e-03, -0.8049510e-04, -0.1115902e-03, -0.1456587e-03, -0.6872960e-03; 
      0.1102029e-03,  1            , -0.1138073e-03,  0.5521437e-04,  0            ,  0            ; 
     -0.2580602e-03, -0.3546250e-03,  1            ,  0            ,  0            ,  0            ;
     -0.2580602e-03, -0.3546250e-03,  0            ,  1            ,  0            ,  0            ;
      0            ,  0            ,  0            ,  0            ,  1            , -0.1122973e-02;
      0            ,  0            ,  0            ,  0            ,  0.1084830e-03,  1            ];

% moment arms 
arm = [0.001923,0.000250,-0.000209,-0.525748,-0.525765,0.000114,-0.262875,0.263078,-0.000975,-1.050331,-1.050106,-1.049434];

% if testSection == 7
%     arm = [0.001923,0.000250,-0.000209,-0.525748,-0.525765,0.000114,-0.262875,0.263078,-0.000975,-1.050331,-1.050106,-1.049434];
% elseif testSection == 9
%     arm = [0.001923,0.000250,-0.000209,-0.525748,-0.525765,0.000114,-0.262875,0.263078,-0.000975,-1.050331,-1.050106,-1.049434];
%     display('Check definition moment arms tunnel 9 -> should they actually be the same for each test section?)
% else
%     error('Unsupported test section number provided. Change input or add test section characteristics to function BAL_getCalFactors.')
% end

% angle offsets
phiYbal = 0; % from tunnel input file CHECK
phiZbal = 0; % from tunnel input file CHECK

% drag force correction parameter to account for influence of FZ
FX_cor = 0; % from tunnel input file

% bending characteristics
x_bend = 0; % from BAL input file
y_bend = 0; % from BAL input file  

% stiffness matrix
e = reshape(load('eij-bal'),12,6);

end % end of function BAL_getCalFactors.m