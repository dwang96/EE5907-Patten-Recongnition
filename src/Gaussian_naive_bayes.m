%% Q2 - Gaussian Naive Bayes
% load data
clear
m_data = load('spamData.mat');
xtrain = m_data.Xtrain;
xtest = m_data.Xtest;
ytrain = m_data.ytrain;
ytest = m_data.ytest;
xtrain = log(xtrain + 0.1);
xtest = log(xtest + 0.1);

%% compute mu and sigma^2
train0 = xtrain(find(ytrain == 0),:);
train1 = xtrain(find(ytrain == 1),:);
mu0 = mean(train0);
sig20 = mean((train0 - mu0).^2);
mu1 = mean(train1);
sig21 = mean((train1 - mu1).^2);
lam_ml = sum(ytrain)/length(ytrain);
[loggau0, loggau1] = gau(xtest, mu0, mu1, sig20, sig21);
classify = (log(lam_ml)+loggau1) - (log(1-lam_ml)+loggau0);
classify(classify>0) = 1;
classify(classify<=0) = 0;
err_test = sum(abs(classify - ytest))/length(ytest);
disp(err_test);

[loggau0, loggau1] = gau(xtrain, mu0, mu1, sig20, sig21);
classify_t = (log(lam_ml)+loggau1) - (log(1-lam_ml)+loggau0);
classify_t(classify_t>0) = 1;
classify_t(classify_t<=0) = 0;
err_train = sum(abs(classify_t - ytrain))/length(ytrain);
disp(err_train);

% Gaussian distribution - 1/sqrt(2*pi*sig2) * exp((x - mu)^2/(2*sig2));
function [loggau0, loggau1] = gau(x, mu0, mu1, sig20, sig21)
loggau0 = zeros(length(x),1);
loggau1 = zeros(length(x),1);
for i = 1:length(x)
    p0_gau = sum(log(exp((-(x(i,:) - mu0).^2)./(2*sig20))./sqrt(2*pi*sig20)));
    p1_gau = sum(log(exp((-(x(i,:) - mu1).^2)./(2*sig21))./sqrt(2*pi*sig21)));
    loggau0(i) = p0_gau;
    loggau1(i) = p1_gau;
end
end