function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 

% 將X(格式mx1)映射成X的1~p次方後存進X_poly矩陣(格式mxp)
% 第一行為X的一次方,故維持原樣
X_poly(:, 1) = X;
% 從2~p的部分,讓X和前一行進行.*的運算
% 按理說計算速度應該能比讓迴圈跑X_poly(:, i) = X .^ i還要來得快吧
for i = 2 : p
  X_poly(:, i) = X .* X_poly(:, i-1);
endfor



% =========================================================================

end
