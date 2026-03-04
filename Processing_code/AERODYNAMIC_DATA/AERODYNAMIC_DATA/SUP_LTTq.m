%% Function SUP_LTTq.m
% Computes freestream dynanic pressure based on input dPb values.
% =========================================================================
% Tomas Sinnige - T.Sinnige@tudelft.nl 
% TU Delft - LR - AWEP - Flight Performance and Propulsion
%
% Version: 1.0
% Last updated:  16 October 2017
% First version: 12 October 2017
% =========================================================================
% | Version |    Date   |   Author  |              Changelog              |
% |---------|-----------|-----------|-------------------------------------|
% |   1.0   | 16/10/'17 | T.Sinnige | Added data tunnel section 9         |
% |---------|-----------|-----------|-------------------------------------|
% |   0.0   | 12/10/'17 | T.Sinnige | First version                       |
% |---------|-----------|-----------|-------------------------------------|
% =========================================================================
% Inputs:  dPb     - pressure difference measured in contraction [Pa]
%          testSec - test section number
% -------------------------------------------------------------------------
% Outputs: qInf    - freestream dynamic pressure [Pa]
% =========================================================================
function qInf = SUP_LTTq(dPb,testSec)

%% Dynamic-pressure calibration data


if testSec == 5 % test section for VGM model
    % limits q(dPb) curve calibration factors
    dPbLim = [100,300]; % [Pa]
   
    % calibration factors q(pB) = facQpB(?,1)*pB^0 + facQpB(?,2)*pB^1 + facQpB(?,3)*pB^2 + facQpB(?,4)*pB^3
    facQpB = [0.51549  2.32312  4.62743e-05  0.0;... %    0      < dPb < dPbLim(1) -> ? = 1 (see eqn above)
              0.51549  2.32312  4.62743e-05  0.0;... % dPbLim(1) < dPb < dPbLim(2) -> ? = 2
              0.51549  2.32312  4.62743e-05  0.0];   %             dPb > dPbLim(2) -> ? = 3

elseif testSec == 7 % test section with ground board
    % limits q(dPb) curve calibration factors
    dPbLim = [50,400]; % [Pa]
    
    % calibration factors q(pB) = facQpB(?,1)*pB^0 + facQpB(?,2)*pB^1 + facQpB(?,3)*pB^2 + facQpB(?,4)*pB^3
    facQpB = [0.909286  2.44320  -3.79098e-05  0.0;... %    0      < dPb < dPbLim(1) -> ? = 1 (see eqn above)
              0.402538  2.45530  -5.74383e-06  0.0;... % dPbLim(1) < dPb < dPbLim(2) -> ? = 2
              3.101380  2.42426   5.22943e-05  0.0];   %             dPb > dPbLim(2) -> ? = 3
elseif testSec == 9 % full section (PIV)
    % limits q(dPb) curve calibration factors
    dPbLim = [100,5000]; % [Pa]
    
    % calibration factors q(pB) = facQpB(?,1)*pB^0 + facQpB(?,2)*pB^1 + facQpB(?,3)*pB^2 + facQpB(?,4)*pB^3
    facQpB = [0.0193846  2.33053  1.72980e-4  0.0;... %    0      < dPb < dPbLim(1) -> ? = 1 (see eqn above)
              0.1903500  2.33534  5.07273e-5  0.0;... % dPbLim(1) < dPb < dPbLim(2) -> ? = 2
              0.1903500  2.33534  5.07273e-5  0.0];   %             dPb > dPbLim(2) -> ? = 3
else
    error('No dynamic-pressure calibration data included yet for this test section.')
end

%% Calibrate data
qInf = zeros(size(dPb));
for i=1:size(dPb,1)
    % determine which calibration polar should be used 
    if dPb < dPbLim(1)
        idxQ = 1;
    elseif dPb < dPbLim(2)
        idxQ = 2;
    else
        idxQ = 3;
    end
    
    % compute the freestream dynamic pressure
    qInf(i) = sum(facQpB(idxQ,:).*[dPb(i)^0,dPb(i)^1,dPb(i)^2,dPb(i)^3]);
end % end for loop over dPb data

end % end of function SUP_LTTq