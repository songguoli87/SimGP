function [gX] = neighborconstraintGradients(model, X)
  
if nargin < 2
  X = model.X;
end

      %%% Prepare to Compute Gradients with respect to X %%%
      gX = zeros(model.N, model.q);
      
      gK = localCovarianceGradients(model);

          %%% Compute Gradients with respect to X %%%
          for i = 1:model.N          
            for j = 1:model.q
                gSX = SXGradXpoint(i,j,model,X);
               gX(i, j) = gX(i, j) + sum(sum( gSX.*gK));
            end
          end

end
 
function gK = localCovarianceGradients(model)
gK = model.lambda * bsxfun(@minus,model.SX,model.y);
% gK = model.lambda*( model.SX - model.y);
end


% function gSX = GradSX(model)
% X = model.X;
% gSX = cell(size(X));
% N = model.N;
% for i = 1:size(X, 1);
%     for j = 1:size(X, 2);
%       gSX{i,j} = SXGradXpoint(i,j,X,N);
%     end
% end
% end  

function gSX = SXGradXpoint(i,j,model,X)
gSX =zeros(size(X,1));
x = X(i,:);
% x1 = x';
% X1 = X';
% n2 = dist2(X,x);
% n2 = distMat(X1,x1);
% wi2 = 0.5 ;
% rbfPart = exp(-n2*wi2);
rbfPart = (model.SX(i,:))';
for k = 1: size(X,1) 
    if (k == i) 
        gSX(:, k) = (X(:, j) - x(j)).*rbfPart;
%         gSX(k, :) = gSX(:, k)';
    end
end
end

