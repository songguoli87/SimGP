function model = constraintExpandParamSim(model,X)

% CONSTRAINTEXPANDPARAMLDA Expands a LDA constraint model
% FORMAT
% DESC Returns expanded model
% ARG model : constraint model
% ARG X : Latent locations
% RETURN model : Returns expanded model
%
% SEEALSO : constraintExpandParam

% DGPLVM


X1 = X;
X2 = X;

SXY = model.SXY;
nPosData = sum(SXY(:));
DXY = 1 - SXY;
nNegData = sum(DXY(:));

 distX = dist2(X1, X2);
%  distX = pdist2(X1, X2,'mahalanobis');

SimX = distX.* SXY;
DifX = (1 - distX).*DXY;


% sigma = model.sigma;
% distX = pdist2(X, X);
% SimX = exp(-0.5*(distX.^2)/sigma);
% NN1 = SimX.*(1 - model.SXY);
% NN2 = SimX.*model.SXY;
% ll =sum(ll(:));
% ll = model.lambda * ll;

model.distX = distX;
model.SimX = SimX;
model.DifX = DifX;
model.nPosData = nPosData;
model.nNegData = nNegData;
return