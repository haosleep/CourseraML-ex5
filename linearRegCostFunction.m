function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% 線性回歸的損失函數和梯度的求法已在第一次作業(ex1)中實踐過
% 正規化部分也沒有什麼差異
% 故不用再多做註解
reg_theta = theta;
reg_theta(1) = 0;

J = sum((X * theta - y) .^ 2) / (2 * m) + (reg_theta' * reg_theta) * lambda / (2 * m);

grad = ((X * theta - y)' * X)' / m + reg_theta * lambda / m;





% =========================================================================

grad = grad(:);

end
