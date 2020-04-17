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

% 這次作業是用線性回歸的資料來練習繪製學習曲線
% 先讀取ex5data1.mat的資料
% 資料內容是由水庫水位的變化來預測水壩的出水量
% 裏面包含了12組訓練資料(X, y 格式12x1)
% 21組驗證資料(Xval, yval 格式21x1)
% 21組測試資料(Xtest, ytest 格式21x1)
load ('ex5data1.mat');

% m = Number of examples
m = size(X, 1);

% Plot training data
% 將訓練資料用二維圖表示出來
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 2: Regularized Linear Regression Cost =============
%  You should now implement the cost function for regularized linear 
%  regression. 
%

% 接著用linearRegCostFunction.m完成加上正規化的線性回歸的損失函數(part2作業)
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

% 這次則是在linearRegCostFunction.m再加上求梯度的功能(part3作業)
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
% 將linearRegCostFunction.m函式完成就可以開始進行機器學習
% 利用trainLinearReg.m訓練得出適合的theta結果
lambda = 0;
[theta] = trainLinearReg([ones(m, 1) X], y, lambda);

%  Plot fit over the data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
% 將訓練後的預測結果用折線繪製在二維圖上
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
% 接著利用learningCurve.m求出訓練集和驗證集的誤差量用以繪製學習曲線(part5作業)

% 順帶一提,這次作業在使用trainLinearReg.m呼叫的fmincg.m會發生除數為0的警告訊息
% 根據課程中的論壇說明
% 這是訓練資料僅有一組或兩組的情況下fmincg.m才會發生的狀況
% 因為實際上的訓練資料不可能這麼少,因此正式使用時按理說也不會發生這種狀況
% (畢竟只有兩組訓練資料,根本連訓練都不用訓練就知道結果不會準了哪還用得著考慮程式碼的問題)
% 且只是警告訊息而已,還是能得出正常的結果,因此可以無視
[error_train, error_val] = ...
    learningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  lambda);

% 再將取得的訓練集和驗證集的誤差量繪製成二維圖
% 從圖中可觀察到當使用的樣本數增加後,訓練集和驗證集都有相當高的誤差量
% 這意味著存在著高偏差的問題
% 因為直線的線性回歸過於簡單造成的欠擬合的結果
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

% part5會發生高偏差(欠擬合)的問題顯而易見是因為直線過於簡單
% 但所持有的資料就僅有一項特徵值而已
% 因此這邊要利用多項式回歸的方式來解決這個問題
p = 8;

% Map X onto Polynomial Features and Normalize
% 用polyFeatures.m將原本的X映射至X的1~p次方(part6作業)
X_poly = polyFeatures(X, p);
% 因為特徵值數量變多了,要用featureNormalize.m進行特徵縮放
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
% 補上X0
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% 驗證集和測試集也要進行同樣的處理
% 使用的平均值和標準差要跟訓練集的相同
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
% 經過特徵縮放後再次進行訓練
[theta] = trainLinearReg(X_poly, y, lambda);

% Plot training data and fit
figure(1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
% 給訓練集的最小值和最大值(用來設定範圍),使用plotFit.m將預測結果繪製在二維圖上
% 因為用來繪製預測結果所用的資料也要進行相同的特徵縮放,故需要傳mu,sigma,p等資訊
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

% 接著建立第二張圖
% 上面繪製學習曲線

% 這邊值得注意的一點是,繪製出來的結果很可能會跟作業的pdf檔中給的答案不一樣
% 根據課程中的論壇說明
% 可能是數值精度導致的問題
% 因為這次作業中,特徵映射到x的8次方,實際上其實很少會取到這麼大的次方數
% 其次是訓練集的數量過少
% 基於這些理由才導致這樣的問題
% 此外,使用的是matlab還是octave,還有使用的版本等等,都可能在這樣的狀況下出現不同的差異
% (其實在polyFeatures.m,用的是.^還是和前一行.*答案就不同了)
% 不過歸根究柢皆是因為8次方的映射和過少的訓練集所導致
% 這也是在正式使用時不該會發生的狀況
% 所以其實最後submit有過的話就也不用太在意這邊的結果跟pdf的標準答案不同就是了
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

% 根據part7說明指示的額外練習,當拉格朗日參數為1或100時候的結果
exLambda = [1 100];
for j = 1:length(exLambda)
  [theta] = trainLinearReg(X_poly, y, exLambda(j));
  % 為了不把前面的圖覆蓋掉的建立新圖
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

% 再來是使用validationCurve.m
% 測試在不同的拉格朗日參數下,學習曲線的結果(part8作業)
[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y, X_poly_val, yval);

close all;
% 最後print出結果
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


% 額外練習
% 上面已經完成了對於各個拉格朗日參數下的曲線結果
% 不過重要的是要確認測試集(全新的資料)的誤差量,才能驗證訓練得出的模型是否適用

% 從上面得到的結果中,根據驗證集的誤差量來決定拉格朗日參數
% bestEVal是最小值,setLidx是第幾項(之前的作業已經用過的方法)
[bestEVal, setLidx] = min(error_val);
% 決定了拉格朗日參數後,用訓練集進行訓練來決定theta
bestTheta = trainLinearReg(X_poly, y, lambda_vec(setLidx));
% 最後驗證測試集的誤差量
error_test = linearRegCostFunction(X_poly_test, ytest, bestTheta, 0);
% 根據教學,應該會在lambda = 3的情況下得到3.8599的結果
fprintf('Theta computed the test error using the best value of lambda you found: \n');
fprintf(' %f \n', error_test);
fprintf('\n');
fprintf('Program paused. Press enter to continue.\n');
pause;


% 額外練習2
% 實用時,特別是對於小型的測試集時
% 在選用i組測試資料時,應從整組測試集中隨機選取i組
% 且在驗證集裡也同樣選取i組
% 由隨機選取出來的測試集和驗證集來計算誤差量
% 以上步驟重複一定次數後,將最終的測試集和驗證集的誤差量求平均
% 這樣得到的結果會比較有幫助

% 根據教學的設定,拉格朗日參數設為0.01,重複次數為50
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
% 因為還得訓練取得theta,可以預見執行完會需要一點時間
% 而且因為是隨機取的,每次結果都會有些許差別所以也難以對答案
% 為了紀念,把第一次跑完的結果在這邊記錄下來
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

% 另外最後輸出的二維圖
% 可以輸入指令"print -dpng '圖片名稱.png'"將結果圖片儲存成png格式