function ll = constraintLogLikelihoodSim(model, X)

% CONSTRAINTLOGLIKELIHOODLDA Constraint loglikelihood for LDA model
% FORMAT
% DESC Returns loglikelihood for constraint
% ARG model : fgplvm model
% ARG X : Latent locations
% RETURN ll : Returns loglikelihood
%
% SEEALSO :
%
% COPYRIGHT : Carl Henrik Ek, 2009

% DGPLVM

% keep non-simliarity
ll_dif =sum(model.DifX((model.DifX > 0)));
ll_dif = model.lambda1*ll_dif ;
% keep simliarity
ll_sim =sum(model.SimX(:));
ll_sim = model.lambda2*ll_sim ;

ll = ll_dif + ll_sim;
ll = -ll;
return;