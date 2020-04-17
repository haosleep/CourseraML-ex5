function plotFit(min_x, max_x, mu, sigma, theta, p)
%PLOTFIT Plots a learned polynomial regression fit over an existing figure.
%Also works with linear regression.
%   PLOTFIT(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
%   fit with power p and feature normalization (mu, sigma).

% Hold on to the current figure
hold on;

% We plot a range slightly bigger than the min and max values to get
% an idea of how the fit will vary outside the range of the data points
% ��X���̤p��-15��X���̤j��+25,�����U�j0.05
% �Ψӧ@��ø�s�w�����G�Ϊ�����X��
x = (min_x - 15: 0.05 : max_x + 25)';

% Map the X values 
% �i��ۦP���S�x�M�g�M�S�x�Y��
X_poly = polyFeatures(x, p);
X_poly = bsxfun(@minus, X_poly, mu);
X_poly = bsxfun(@rdivide, X_poly, sigma);

% Add ones
X_poly = [ones(size(x, 1), 1) X_poly];

% Plot
% ø�s���G
plot(x, X_poly * theta, '--', 'LineWidth', 2)

% Hold off to the current figure
hold off

end
