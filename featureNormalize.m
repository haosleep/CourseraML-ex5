function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% 這邊是利用了bsxfun來進行特徵縮放
% 會比跑迴圈逐列運算還要來得有效率
% bsxfun的句法是bsxfun(@fun, A, B)
% 所使用的fun必須是二進制函數(只能有兩個輸入)
% AB矩陣維度必須要能相對應
% (格式相同(都是axb),或是不同的格式其中一項是1(像是一個axb一個ax1這樣))

% 輸入的X是用polyFeatures.m映射後的X_poly(格式是12x9)
% 取每一行的平均值存於mu(格式1x9)
% @minus指的是-的函數
% 因為滿足條件,mu會直接將矩陣擴展成對應的12x9後讓X_poly - mu
mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

% @rdivide則是./的函數
sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);

% 不過在第一次作業(ex1中的featureNormalize.m)時即使不用bsxfun也能做到一樣的結果
% 不確定哪種方法比較省效能就是了
% 也或許是使用的是Octave 5.2版才能讓12x9和1x9的矩陣直接相減或點除也說不定

% ============================================================

end
