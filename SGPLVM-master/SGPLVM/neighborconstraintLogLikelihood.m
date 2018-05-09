function ll = neighborconstraintLogLikelihood(model,X)

% CONSTRAINTLOGLIKELIHOODLDA Constraint loglikelihood for neighbor
% FORMAT
% DESC Returns loglikelihood for constraint
% ARG model : fgplvm model
% ARG X : Latent locations
% RETURN ll : Returns loglikelihood
%
% SEEALSO :
%
% COPYRIGHT : Carl Henrik Ek, 2009

if nargin < 2
  X = model.X;
end
% H = model.y - model.SX;
% ll = 0.5*norm(H,'fro');
H = bsxfun(@minus,model.y,model.SX);
ll = sqrt(sum(H(:).^2));
ll = model.lambda*ll;
ll = -ll;
return;