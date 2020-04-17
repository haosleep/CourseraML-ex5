%% Machine Learning Online Class
%  Exercise 5 | Regularized Linear Regression and Bias-Variance
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     linearRegCostFunction.m
%     learningCurve.m
%     validationCurve.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  The following code will load the dataset into your environment and plot
%  the data.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment

% �o���@�~�O�νu�ʦ^�k����ƨӽm��ø�s�ǲߦ��u
% ��Ū��ex5data1.mat�����
% ��Ƥ��e�O�Ѥ��w���쪺�ܤƨӹw�����򪺥X���q
% �ح��]�t�F12�հV�m���(X, y �榡12x1)
% 21�����Ҹ��(Xval, yval �榡21x1)
% 21�մ��ո��(Xtest, ytest �榡21x1)
load ('ex5data1.mat');

% m = Number of examples
m = size(X, 1);

% Plot training data
% �N�V�m��ƥΤG���Ϫ�ܥX��
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 2: Regularized Linear Regression Cost =============
%  You should now implement the cost function for regularized linear 
%  regression. 
%

% ���ۥ�linearRegCostFunction.m�����[�W���W�ƪ��u�ʦ^�k���l�����(part2�@�~)
theta = [1 ; 1];
J = linearRegCostFunction([ones(m, 1) X], y, theta, 1);

fprintf(['Cost at theta = [1 ; 1]: %f '...
         '\n(this value should be about 303.993192)\n'], J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 3: Regularized Linear Regression Gradient =============
%  You should now implement the gradient for regularized linear 
%  regression.
%

% �o���h�O�blinearRegCostFunction.m�A�[�W�D��ת��\��(part3�@�~)
theta = [1 ; 1];
[J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, 1);

fprintf(['Gradient at theta = [1 ; 1]:  [%f; %f] '...
         '\n(this value should be about [-15.303016; 598.250744])\n'], ...
         grad(1), grad(2));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =========== Part 4: Train Linear Regression =============
%  Once you have implemented the cost and gradient correctly, the
%  trainLinearReg function will use your cost function to train 
%  regularized linear regression.
% 
%  Write Up Note: The data is non-linear, so this will not give a great 
%                 fit.
%

%  Train linear regression with lambda = 0
% �NlinearRegCostFunction.m�禡�����N�i�H�}�l�i������ǲ�
% �Q��trainLinearReg.m�V�m�o�X�A�X��theta���G
lambda = 0;
[theta] = trainLinearReg([ones(m, 1) X], y, lambda);

%  Plot fit over the data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
% �N�V�m�᪺�w�����G�Χ�uø�s�b�G���ϤW
plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
hold off;

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =========== Part 5: Learning Curve for Linear Regression =============
%  Next, you should implement the learningCurve function. 
%
%  Write Up Note: Since the model is underfitting the data, we expect to
%                 see a graph with "high bias" -- Figure 3 in ex5.pdf 
%

lambda = 0;
% ���ۧQ��learningCurve.m�D�X�V�m���M���Ҷ����~�t�q�ΥHø�s�ǲߦ��u(part5�@�~)

% ���a�@��,�o���@�~�b�ϥ�trainLinearReg.m�I�s��fmincg.m�|�o�Ͱ��Ƭ�0��ĵ�i�T��
% �ھڽҵ{�����׾»���
% �o�O�V�m��ƶȦ��@�թΨ�ժ����p�Ufmincg.m�~�|�o�ͪ����p
% �]����ڤW���V�m��Ƥ��i��o���,�]�������ϥήɫ��z���]���|�o�ͳo�ت��p
% (�����u����հV�m���,�ڥ��s�V�m�����ΰV�m�N���D���G���|�ǤF���٥αo�ۦҼ{�{���X�����D)
% �B�u�Oĵ�i�T���Ӥw,�٬O��o�X���`�����G,�]���i�H�L��
[error_train, error_val] = ...
    learningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  lambda);

% �A�N���o���V�m���M���Ҷ����~�t�qø�s���G����
% �q�Ϥ��i�[����ϥΪ��˥��ƼW�[��,�V�m���M���Ҷ������۷����~�t�q
% �o�N���ۦs�b�۰����t�����D
% �]�����u���u�ʦ^�k�L��²��y���������X�����G
plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 6: Feature Mapping for Polynomial Regression =============
%  One solution to this is to use polynomial regression. You should now
%  complete polyFeatures to map each example into its powers
%

% part5�|�o�Ͱ����t(�����X)�����D��ө����O�]�����u�L��²��
% ���ҫ�������ƴN�Ȧ��@���S�x�ȦӤw
% �]���o��n�Q�Φh�����^�k���覡�ӸѨM�o�Ӱ��D
p = 8;

% Map X onto Polynomial Features and Normalize
% ��polyFeatures.m�N�쥻��X�M�g��X��1~p����(part6�@�~)
X_poly = polyFeatures(X, p);
% �]���S�x�ȼƶq�ܦh�F,�n��featureNormalize.m�i��S�x�Y��
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
% �ɤWX0
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% ���Ҷ��M���ն��]�n�i��P�˪��B�z
% �ϥΪ������ȩM�зǮt�n��V�m�����ۦP
% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

fprintf('Normalized Training Example 1:\n');
fprintf('  %f  \n', X_poly(1, :));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;



%% =========== Part 7: Learning Curve for Polynomial Regression =============
%  Now, you will get to experiment with polynomial regression with multiple
%  values of lambda. The code below runs polynomial regression with 
%  lambda = 0. You should try running the code with different values of
%  lambda to see how the fit and learning curve change.
%

lambda = 0;
% �g�L�S�x�Y���A���i��V�m
[theta] = trainLinearReg(X_poly, y, lambda);

% Plot training data and fit
figure(1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
% ���V�m�����̤p�ȩM�̤j��(�Ψӳ]�w�d��),�ϥ�plotFit.m�N�w�����Gø�s�b�G���ϤW
% �]���Ψ�ø�s�w�����G�ҥΪ���Ƥ]�n�i��ۦP���S�x�Y��,�G�ݭn��mu,sigma,p����T
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

% ���۫إ߲ĤG�i��
% �W��ø�s�ǲߦ��u

% �o��ȱo�`�N���@�I�O,ø�s�X�Ӫ����G�ܥi��|��@�~��pdf�ɤ��������פ��@��
% �ھڽҵ{�����׾»���
% �i��O�ƭȺ�׾ɭP�����D
% �]���o���@�~��,�S�x�M�g��x��8����,��ڤW���ܤַ|����o��j�������
% �䦸�O�V�m�����ƶq�L��
% ���o�ǲz�Ѥ~�ɭP�o�˪����D
% ���~,�ϥΪ��Omatlab�٬Ooctave,�٦��ϥΪ���������,���i��b�o�˪����p�U�X�{���P���t��
% (���bpolyFeatures.m,�Ϊ��O.^�٬O�M�e�@��.*���״N���P�F)
% ���L�k�ڨs�k�ҬO�]��8���誺�M�g�M�L�֪��V�m���ҾɭP
% �o�]�O�b�����ϥήɤ��ӷ|�o�ͪ����p
% �ҥH���̫�submit���L���ܴN�]���ΤӦb�N�o�䪺���G��pdf���зǵ��פ��P�N�O�F
figure(2);
[error_train, error_val] = ...
    learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')

fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

% �ھ�part7�������ܪ��B�~�m��,��Ԯ�Ԥ�ѼƬ�1��100�ɭԪ����G
exLambda = [1 100];
for j = 1:length(exLambda)
  [theta] = trainLinearReg(X_poly, y, exLambda(j));
  % ���F����e�������л\�����إ߷s��
  figure(j * 2 + 1);
  plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
  plotFit(min(X), max(X), mu, sigma, theta, p);
  xlabel('Change in water level (x)');
  ylabel('Water flowing out of the dam (y)');
  title (sprintf('Polynomial Regression Fit (lambda = %f)', exLambda(j)));
  figure(j * 2 + 2);
  [error_train, error_val] = ...
      learningCurve(X_poly, y, X_poly_val, yval, exLambda(j));
  plot(1:m, error_train, 1:m, error_val);

  title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', exLambda(j)));
  xlabel('Number of training examples')
  ylabel('Error')
  axis([0 13 0 100])
  legend('Train', 'Cross Validation')

  fprintf('Polynomial Regression (lambda = %f)\n\n', exLambda(j));
  fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
  for i = 1:m
      fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
  end

  fprintf('Program paused. Press enter to continue.\n');
  pause;
endfor


%% =========== Part 8: Validation for Selecting Lambda =============
%  You will now implement validationCurve to test various values of 
%  lambda on a validation set. You will then use this to select the
%  "best" lambda value.
%

% �A�ӬO�ϥ�validationCurve.m
% ���զb���P���Ԯ�Ԥ�ѼƤU,�ǲߦ��u�����G(part8�@�~)
[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y, X_poly_val, yval);

close all;
% �̫�print�X���G
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;


% �B�~�m��
% �W���w�g�����F���U�өԮ�Ԥ�ѼƤU�����u���G
% ���L���n���O�n�T�{���ն�(���s�����)���~�t�q,�~�����ҰV�m�o�X���ҫ��O�_�A��

% �q�W���o�쪺���G��,�ھ����Ҷ����~�t�q�ӨM�w�Ԯ�Ԥ�Ѽ�
% bestEVal�O�̤p��,setLidx�O�ĴX��(���e���@�~�w�g�ιL����k)
[bestEVal, setLidx] = min(error_val);
% �M�w�F�Ԯ�Ԥ�Ѽƫ�,�ΰV�m���i��V�m�ӨM�wtheta
bestTheta = trainLinearReg(X_poly, y, lambda_vec(setLidx));
% �̫����Ҵ��ն����~�t�q
error_test = linearRegCostFunction(X_poly_test, ytest, bestTheta, 0);
% �ھڱо�,���ӷ|�blambda = 3�����p�U�o��3.8599�����G
fprintf('Theta computed the test error using the best value of lambda you found: \n');
fprintf(' %f \n', error_test);
fprintf('\n');
fprintf('Program paused. Press enter to continue.\n');
pause;


% �B�~�m��2
% ��ή�,�S�O�O���p�������ն���
% �b���i�մ��ո�Ʈ�,���q��մ��ն����H�����i��
% �B�b���Ҷ��̤]�P�˿��i��
% ���H������X�Ӫ����ն��M���Ҷ��ӭp��~�t�q
% �H�W�B�J���Ƥ@�w���ƫ�,�N�̲ת����ն��M���Ҷ����~�t�q�D����
% �o�˱o�쪺���G�|��������U

% �ھڱоǪ��]�w,�Ԯ�Ԥ�ѼƳ]��0.01,���Ʀ��Ƭ�50
exRanLambda = 0.01
exRanM = size(X_poly, 1);
exRanValM = size(X_poly_val, 1);
exRanErrTra = zeros(exRanM, 1);
exRanErrVal = zeros(exRanM, 1);
for i = 1 : exRanM
  for j = 1 : 50
    exRanIdx = randperm(exRanM);
    exRanValIdx = randperm(exRanValM);
    exRanSetTheta = trainLinearReg(X_poly(exRanIdx(1:i), :), y(exRanIdx(1:i)), exRanLambda);
    exRanErrTra(i) = exRanErrTra(i) + ...
       linearRegCostFunction(X_poly(exRanIdx(1:i), :), y(exRanIdx(1:i)), exRanSetTheta, 0);
    exRanErrVal(i) = exRanErrVal(i) + ...
       linearRegCostFunction(X_poly_val(exRanValIdx(1:i), :), yval(exRanValIdx(1:i)), exRanSetTheta, 0);
  endfor
  exRanErrTra(i) = exRanErrTra(i) / 50;
  exRanErrVal(i) = exRanErrVal(i) / 50;  
endfor

plot(1:exRanM, exRanErrTra, 1:exRanM, exRanErrVal);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', exRanLambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')

fprintf('Polynomial Regression (lambda = %f)\n\n', exRanLambda);
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, exRanErrTra(i), exRanErrVal(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;
% �]���ٱo�V�m���otheta,�i�H�w�����槹�|�ݭn�@�I�ɶ�
% �ӥB�]���O�H������,�C�����G���|���ǳ\�t�O�ҥH�]���H�ﵪ��
% ���F����,��Ĥ@���]�������G�b�o��O���U��
% Training Examples     Train Error     Cross Validation Error
%       1               0.000000        126.162029
%       2               0.007465        77.802296
%       3               0.025495        48.832771
%       4               0.025393        32.549670
%       5               0.053510        16.910197
%       6               0.066431        20.275142
%       7               0.099746        18.343237
%       8               0.121369        15.676106
%       9               0.159475        18.235299
%       10              0.176526        15.972716
%       11              0.199196        13.755519
%       12              0.222173        15.502436

% �t�~�̫��X���G����
% �i�H��J���O"print -dpng '�Ϥ��W��.png'"�N���G�Ϥ��x�s��png�榡