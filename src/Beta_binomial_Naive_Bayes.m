%% Q1 - Beta-binomial Naive Bayes
%% Data Readin
clear
load('spamData.mat'); % load the data - please remember to add the data into this folder to run the code
xtrain = Xtrain;        % train features
xtest = Xtest;          % test features
% 1 = spam; 0 = non-spam
num_features = size(xtrain,2);


%% data process -  bin
xtrain(xtrain > 0) = 1;
xtest(xtest > 0)= 1;

%% step 1: compute lam_ml and eta_map
lam_ml = sum(ytrain)/length(ytrain);
n_y1 = sum(ytrain);
n_y0 = length(ytrain) - n_y1;
alpha = 0:0.5:100;
% compute p(xi = 0 or 1|D) for where y = 1 or 0 means spam
idx_1 = find(ytrain == 1);
train1_set = xtrain(idx_1,:);
idx_0 = find(ytrain == 0);
train0_set = xtrain(idx_0,:);
n11 = sum(train1_set,1);        % y = 1 and xij = 1
n10 = length(train1_set) - n11; % y = 1 and xij = 0
n01 = sum(train0_set,1);        % y = 0 and xij = 1
n00 = length(train0_set) - n01; % y = 0 and xij = 0
err_train = zeros(length(alpha),1);
err_test = zeros(length(alpha),1);
for i = 1:length(alpha)
    [logpy1, logpy0] = posterior(n00, n01, n10, n11, n_y0, n_y1, xtrain, alpha(i));
    result_1 = log(lam_ml) + logpy1;
    result_0 = log(1 - lam_ml) + logpy0;
    result_train = result_1 - result_0;
    result_train(result_train>0) = 1;
    result_train(result_train<0) = 0;
    err_tr = abs(result_train - ytrain);
    err_train(i) = sum(err_tr) / length(ytrain);
end

for i = 1:length(alpha)
    [logpy1_e, logpy0_e] = posterior(n00, n01, n10, n11, n_y0, n_y1, xtest, alpha(i));
    result_1_e = log(lam_ml) + logpy1_e;
    result_0_e = log(1 - lam_ml) + logpy0_e;
    result_test = result_1_e - result_0_e;
    result_test(result_test>0) = 1;
    result_test(result_test<0) = 0;
    err_te = abs(result_test - ytest);
    err_test(i) = sum(err_te) / length(ytest);
end
plot(alpha,err_train)
hold on
plot(alpha,err_test)
xlabel('\alpha')
ylabel('error rate')
legend('error_{train}','error_{test}','Location','northwest')
disp(err_train(3))
disp(err_train(21))
disp(err_train(201))
disp(err_test(3))
disp(err_test(21))
disp(err_test(201))
    
function [logpy1, logpy0] = posterior(n00, n01, n10, n11, n_y0, n_y1, x, a)

logpy1 = zeros(length(x), 1);
logpy0 = zeros(length(x), 1);
for j = 1:length(x)
    py1 = 0;
    py0 = 0;
    for k = 1:size(x,2)
        if (x(j,k) == 1)
            py1 = py1 + log((n11(k) + a) / (n_y1 + 2*a));
            py0 = py0 + log((n01(k) + a) / (n_y0 + 2*a));
        else
            py1 = py1 + log((n10(k) + a) / (n_y1 + 2*a));
            py0 = py0 + log((n00(k) + a) / (n_y0 + 2*a));
        end
    end
    logpy1(j) = py1;
    logpy0(j) = py0;
end
end