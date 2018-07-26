function makedata_pascal

disp('Generating similarities and neighbourhood.');
load('./data/pascal1K.mat');

feax_tr = I_tr;
feay_tr = T_tr;
feax_te = I_te;
feay_te = T_te;
clear I_tr T_tr I_te T_te;

sigma = 1;

% for X
[EX_tr] = pdist2(feax_tr, feax_tr,'euclidean');
SX_tr = exp(-0.5*(EX_tr.^2)/sigma);%UCSD_SIMEUC%d
simx = prctile(SX_tr(:),85);% use top 15 as NN
[EX_te] = pdist2(feax_te, feax_tr,'euclidean');
SX_te = exp(-0.5*(EX_te.^2)/sigma);
NX_tr = (SX_tr)>simx;
NX_te = (SX_te)>simx;

% for Y
[EY_tr] = pdist2(feay_tr, feay_tr,'euclidean');  
SY_tr = exp(-0.5*(EY_tr.^2)/sigma);%UCSD_SIMEUC%d
simy = prctile(SY_tr(:),85);% use top 15 as NN
[EY_te] = pdist2(feay_te, feay_tr,'euclidean');
SY_te = exp(-0.5*(EY_te.^2)/sigma);
NY_tr = (SY_tr)>simy;
NY_te = (SY_te)>simy;

% for XY
SXY_tr = (NX_tr + NY_tr)>0;
SXY_te = (NX_te + NY_te)>0;    

 % use flitered sim files
 SX_tr = SX_tr.*SXY_tr;% non-neighbor points have similarity 0
 SY_tr = SY_tr.*SXY_tr;
 SX_te = SX_te.*SXY_te;
 SY_te = SY_te.*SXY_te;

save(sprintf('./data/pascal_sim_euc85.mat'),'SX_tr', 'SY_tr', 'SX_te', 'SY_te');
fprintf('Done.\n');
end
