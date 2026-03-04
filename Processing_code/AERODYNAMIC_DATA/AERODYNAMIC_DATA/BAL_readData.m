%% Function BAL_readData.m
% Reads LTT balance data files
% =========================================================================
% Tomas Sinnige - T.Sinnige@tudelft.nl 
% TU Delft - LR - AWEP - Flight Performance and Propulsion
%
% Version: 3.0
% Last updated:  17 January 2021
% First version: 26 May 2017
% =========================================================================
% | Version |    Date   |   Author  |              Changelog              |
% |---------|-----------|-----------|-------------------------------------|
% |   3.0   | 17/01/'21 |  P.Lopez  | -) Changed data into structure type |
% |         |           |   (TA)    |                                     |
% |---------|-----------|-----------|-------------------------------------|
% |   2.0   | 30/11/'18 | T.Sinnige | -) Outlier filtering now based on   |
% |         |           |           |    input variable dPbCut            |
% |---------|-----------|-----------|-------------------------------------|
% |   1.0   | 16/10/'17 | T.Sinnige | -) Removed averaging over repeated  |
% |         |           |           |    operating conditions             |
% |         |           |           | -) Moved loading zero-measurement   |
% |         |           |           |    data to separate function        |
% |         |           |           | -) Removed extra raw field from     |
% |         |           |           |    output structure                 |
% |---------|-----------|-----------|-------------------------------------|
% |   0.0   | 26/05/'17 | T.Sinnige | First version                       |
% |---------|-----------|-----------|-------------------------------------|
% =========================================================================
% Inputs:  fn     - filename of the raw measurement file
%          idxB   - indices in Balance data structures
%          dPbCut - cut-off dPb for filtering of outliers 
% -------------------------------------------------------------------------
% Outputs: data - structure containing output data
%                  
% =========================================================================
function DATA = BAL_readData(fn,idxB)

% %% Print status update
% display(['Loading balance data: ',fn])
% data.fn = fn

%% Load Data
% raw data
fid      = fopen(fn);
read_data = cell2mat(textscan(fid,['%f %f:%f:%f',repmat(' %f',1,31)],'headerlines',2));
fclose(fid);

%% Insert data into structure 
DATA.run   = read_data(:,idxB.run);
DATA.hr    = read_data(:,idxB.hr);
DATA.min   = read_data(:,idxB.min);
DATA.sec   = read_data(:,idxB.sec);
DATA.AoA   = read_data(:,idxB.AoA);
DATA.AoS   = read_data(:,idxB.AoS);
DATA.dPb   = read_data(:,idxB.dPb);
DATA.pBar  = read_data(:,idxB.pBar);
DATA.temp  = read_data(:,idxB.temp);
DATA.B     = read_data(:,idxB.B);
DATA.B1    = read_data(:,idxB.B1);
DATA.B2    = read_data(:,idxB.B2);
DATA.B3    = read_data(:,idxB.B3);
DATA.B4    = read_data(:,idxB.B4);
DATA.B5    = read_data(:,idxB.B5);
DATA.B6    = read_data(:,idxB.B6);
DATA.rpmWT = read_data(:,idxB.rpmWT);
DATA.rho   = read_data(:,idxB.rho);
DATA.q     = read_data(:,idxB.q);
DATA.V     = read_data(:,idxB.V);
DATA.Re    = read_data(:,idxB.Re);
DATA.rpsM1 = read_data(:,idxB.rpsM1);
DATA.rpsM2 = read_data(:,idxB.rpsM2);
DATA.iM1   = read_data(:,idxB.iM1);
DATA.iM2   = read_data(:,idxB.iM2);
DATA.dPtQ  = read_data(:,idxB.dPtQ);

DATA.tM1   = read_data(:,idxB.tM1);
DATA.tM2   = read_data(:,idxB.tM2);
DATA.vM1   = read_data(:,idxB.vM1);
DATA.vM2   = read_data(:,idxB.vM2);

% DATA.FX = read_data(:,idxB.FX);
% DATA.FY = read_data(:,idxB.FY);
% DATA.FZ = read_data(:,idxB.FZ);
% DATA.MX = read_data(:,idxB.MX);
% DATA.MY = read_data(:,idxB.MY);
% DATA.MZ = read_data(:,idxB.MZ);
% DATA.CFX = read_data(:,idxB.CFX);
% DATA.CFY = read_data(:,idxB.CFY);
% DATA.CFZ = read_data(:,idxB.CFZ);
% DATA.CMX = read_data(:,idxB.CMX);
% DATA.CMY = read_data(:,idxB.CMY);
% DATA.CMZ = read_data(:,idxB.CMZ);
% DATA.N = read_data(:,idxB.N);
% DATA.T = read_data(:,idxB.T);
% DATA.Y = read_data(:,idxB.Y);
% DATA.L = read_data(:,idxB.L);
% DATA.D = read_data(:,idxB.D);
% DATA.Mr = read_data(:,idxB.Mr);
% DATA.Mp = read_data(:,idxB.Mp);
% DATA.My = read_data(:,idxB.My);
% DATA.CN = read_data(:,idxB.CN);
% DATA.CT = read_data(:,idxB.CT);
% DATA.CY = read_data(:,idxB.CY);
% DATA.CL = read_data(:,idxB.CL);
% DATA.CD = read_data(:,idxB.CD);
% DATA.CMr = read_data(:,idxB.CMr);
% DATA.CMp = read_data(:,idxB.CMp);
% DATA.CMy = read_data(:,idxB.CMy);
% DATA.CTp = read_data(:,idxB.CTp);
% DATA.TCp = read_data(:,idxB.TCp);
% DATA.TCw = read_data(:,idxB.TCw);
% DATA.qInf_pt = read_data(:,idxB.qInf_pt);
% DATA.J = read_data(:,idxB.J);
% DATA.Jori = read_data(:,idxB.Jori);
% DATA.CLp = read_data(:,idxB.CLp);
% DATA.CDp = read_data(:,idxB.CDp);
% DATA.CLwoP = read_data(:,idxB.CLwoP);
% DATA.CDwoP = read_data(:,idxB.CDwoP);
% DATA.CLempty = read_data(:,idxB.CLempty);
% DATA.CDempty = read_data(:,idxB.CDempty);
% DATA.b = read_data(:,idxB.b);
% DATA.c = read_data(:,idxB.c);
% DATA.S = read_data(:,idxB.S);
% DATA.nu = read_data(:,idxB.nu);
% DATA.pInf = read_data(:,idxB.pInf);
% DATA.CLori = read_data(:,idxB.CLori);
% DATA.CDori = read_data(:,idxB.CDori);
% DATA.dPtM = read_data(:,idxB.dPtM);
% DATA.dPsInf = read_data(:,idxB.dPsInf);
% DATA.CYaw = read_data(:,idxB.CYaw);
% DATA.CMp25c = read_data(:,idxB.CMp25c);

%DATA.Hour=
% if ~isempty(data.unc)
%     data.unc(idxOutlier,:) = [];
% end
% if ~isempty(data.cor)
%     data.cor(idxOutlier,:) = [];
% end

end % end of function BAL_readData.m