clear all;
close all;
% addpath(genpath('F:\experiments\SGPLVM-master\SGPLVM-master\matlab'));

%%  Load Data

dataType = 'pascal';
dataSetNames = 'pascal';
numberOfDatasets=2;
load('data/pascal1k_similarity_euc.mat');

Y_train = SX_tr_euc;
Z_train = SY_tr_euc;
Yts{1} = SX_te_euc;
Yts{2} = SY_te_euc;
clear SX_tr_euc SY_tr_euc SX_te_euc SY_te_euc;


load('data/pascal1K_cat');
L = length(cat.tr);
for i=1:L
    for j=1:L
        SXY_tr(i,j) = (cat.tr(i)==cat.tr(j));
    end
end

if(size(Y_train,1)>100)
  approx = 'fitc'; %fully independent training conditional
else
  approx = 'ftc'; % no approximation
end

%%  Learn Initialisation through CCA
% 
load('data/pascal1K.mat');
 
fprintf('performing svd...\n');

[~,~,v] = svds(I_tr,128);
X = I_tr * v;
[~,~,v] = svds(T_tr,10);
Y = T_tr * v;
    
% X = I_tr;
% Y = T_tr;
clear I_tr T_tr I_te T_te;

Kx = X*X';
Ky = Y*Y';
clear X Y;

% pre-process Kernels
Kx = kernelCenter(Kx);
Ky = kernelCenter(Ky);
Kx = Kx./sum(diag(Kx));
Ky = Ky./sum(diag(Ky));
Kx = (Kx+Kx')./2;
Ky= (Ky+Ky')./2;
   
[A,B] = canoncorr(Kx,Ky);
Xcca = Kx*A;
Ycca = Ky*B;

Xs = (1/2).*(Xcca+Ycca);
X_init = Xs ;
X_init = (X_init-repmat(mean(X_init),size(X_init,1),1))./repmat(std(X_init),size(X_init,1),1);
clear Kx Ky Xcca Ycca A B;

q = size(X_init,2);

%% Create SGPLVM model
  options_y = fgplvmOptions(approx);
  options_y.optimiser = 'scg2';
  options_y.scale2var1 = true;
  options_y.initX = X_init;
  options_y.prior = []; % simGP
  model{1} = fgplvmCreate(size(options_y.initX,2),size(Y_train,2),Y_train,options_y);
  
  options_z = fgplvmOptions(approx);
  options_z.optimiser = 'scg2';
  options_z.scale2var1 = true;
  options_z.initX = X_init;
  options_z.prior = []; % simGP
  model{2} = fgplvmCreate(size(options_z.initX,2),size(Z_train,2),Z_train,options_z);

  options = sgplvmOptions;
  options.save_intermediate = inf;
  options.name = 'sgplvm_cca_test_';
  options.initX = zeros(2,size(X_init,2));
  options.initX(1,:) = true;
  options.initX(2,:) = true;
  
  options.neighborconstraints = true; % dsimgp
  options.lambda = 1e0;
  options.gamma = 1e0;
  
   model = sgplvmCreate(model,[],options);
   
  options_constraint = constraintOptions('Sim'); % rsimgp
  options_constraint.lambda1 = 1e0;
  options_constraint.lambda2 = 1e0;
  options_constraint.N = model.N;
  options_constraint.q = model.q;
  options_constraint.dim = 1:model.q;
  options_constraint.SXY = SXY_tr;
  model = sgplvmAddConstraint(model,options_constraint);
   
  %%  Train SGPLVM model
 nr_iters = 500;
 model = sgplvmOptimise(model,true,nr_iters,false,false);
save(sprintf('drsimgp_pascal_model.mat'), '-v7.3','model');

%%  Test SGPLVM model
obsMod = 1; % one of the involved sub-models (the one for which we have the data)
infMod = setdiff(1:2, obsMod);
numberTestPoints = size(Yts{obsMod},1);
perm = randperm(size(Yts{obsMod},1));
testInd = perm(1:numberTestPoints);

L = length(cat.te);
% image query
Zpred = zeros(length(testInd), size(model.comp{infMod}.y,2));
for i=1:length(testInd)
    curInd = testInd(i);
    fprintf('# Testing indice number %d ', curInd);

    fprintf('taken from the image test set\n');
    y_star = Yts{obsMod}(curInd,:);
    index_in = 1;
    index_out = setdiff(1:2, index_in);        
    x_star = sgplvmPointOut(model,index_in,index_out,y_star);         
    XZpred(curInd,:) = x_star;
end  
 save(sprintf('XZpred_pascal_drsimgp.mat'), 'XZpred');
 
%% text query
Ypred = zeros(length(testInd), size(model.comp{obsMod}.y,2));
for i=1:length(testInd)
    curInd = testInd(i);
    fprintf('# Testing indice number %d ', curInd);

    fprintf('taken from the text test set\n');
    z_star = Yts{infMod}(curInd,:);
    index_in = 2;
    index_out = setdiff(1:2, index_in);        
    x_star = sgplvmPointOut(model,index_in,index_out,z_star);       
    XYpred(curInd,:) = x_star;
end  
save(sprintf('XYpred_pascal_drsimgp.mat'), 'XYpred');
fprintf(' Finish testing.\n')
    
%% cross-modal retrieval     

opt.metric='NC';
opt.rm=0;
fprintf('----------------------------------------------\n');
fprintf(' Image queries (retrieve texts)\n');
[Q,C,im2txt] = retrieval(XZpred,cat.te,XYpred,cat.te,opt);
im2txt

fprintf('\n\n----------------------------------------------\n');
fprintf(' Text queries (retrieve images)\n');
[Q,C,txt2im] = retrieval(XYpred,cat.te,XZpred,cat.te,opt);
txt2im
save(sprintf('results/retrieval_drsimgp_%d.mat', q), 'im2txt', 'txt2im');

fprintf(' done.\n')
