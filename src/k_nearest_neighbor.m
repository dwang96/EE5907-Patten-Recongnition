%% Q4 - K-Nearest Neighbors
clear
load('spamData.mat');
xtrain = Xtrain;
xtest = Xtest;
xtrain = log(xtrain + 0.1); % 3065 samples, feature length 57.
xtest = log(xtest + 0.1);   % 1536 samples, feature length 57.
k = [1:9,10:5:100];         % according to the question
e_test = zeros(1, length(k));
e_train = zeros(1, length(k));
for n = 1:length(k)
    [err_test, err_train] = mKNN(xtrain, ytrain, xtest, ytest, k(n));
    e_test(n) = err_test;
    e_train(n) = err_train;
end
plot(k, e_train, '-o')
hold on
plot(k, e_test, '-x')
legend('error rate_{train}','error rate_{test}')
xlabel('k')
ylabel('error rate')

function [err_test, err_train] = mKNN(xtrain, ytrain, xtest, ytest, k)
dist_tr = zeros(length(xtrain));
dist_te = zeros(length(xtest), length(xtrain));
for i = 1:length(xtest)
    for j = 1:length(xtrain)
        dist_te(i,j) = norm(xtest(i,:) - xtrain(j,:));
    end
end
for i = 1:length(xtrain)
    for j = 1:length(xtrain)
        dist_tr(i,j) = norm(xtrain(i,:) - xtrain(j,:));
    end
end

%% find closest k neighbor
n_xtrain = zeros(length(xtrain),k);
n_xtest = zeros(length(xtest),k);
for i = 1:length(xtrain)
    [~, n_tr] = sort(dist_tr(i,:));
    n_xtrain(i,:) = n_tr(1:k);
end
for i = 1:length(xtest)
    [~, n_te] = sort(dist_te(i,:));
    n_xtest(i,:) = n_te(1:k);
end

p1_tr = sum(ytrain(n_xtrain),2)/k;
p1_te = sum(ytrain(n_xtest),2)/k;
p0_tr = 1 - p1_tr;
p0_te = 1 - p1_te;
ytr = zeros(length(xtrain),1);
yte = zeros(length(xtest),1);
ytr(find(p1_tr - p0_tr >=0),:) = 1;
yte(find(p1_te - p0_te >=0),:) = 1;
err_train = sum(abs(ytr - ytrain))/length(xtrain);
err_test = sum(abs(yte - ytest))/length(xtest);
fprintf('error_rate_test=%f\n',err_test);
fprintf('error_rate_train=%f\n',err_train);
end