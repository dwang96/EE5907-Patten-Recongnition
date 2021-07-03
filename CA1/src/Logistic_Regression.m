clear
load('spamData.mat');
features = length(Xtrain(1,:));
train_num =  length(Xtrain(:,1));
test_num = length(Xtest(:,1));
xtrain = log(Xtrain + 0.1);
xtest = log(Xtest + 0.1);
xtrain2 = ones(train_num,features+1);
xtest2 = ones(test_num,features+1);
xtrain2(:,2:features+1) = xtrain;
xtest2(:,2:features+1) = xtest;
lam = [1:9,10:5:100];            % lambda for regularization
e_test = zeros(1, length(lam));
e_train = zeros(1, length(lam));
for i = 1:length(lam)
    [err_train, err_test] = logistic_reg(lam(i), xtrain2, xtest2, ytrain, ytest, features);
    e_test(i) = err_test;
    e_train(i) = err_train;
end
plot(lam, e_train, '-o')
hold on
plot(lam, e_test, '-x')
legend('error rate_{train}','error rate_{test}','Location','northwest')
xlabel('\lambda')
ylabel('error rate')

%%
function [err_train, err_test] = logistic_reg(lam, xtrain2, xtest2, ytrain, ytest, features)
w = zeros(features + 1, 1);
w1 = w;
w1(1) = 0;
I = diag(ones(1,features +1));
I(1,1) = 0;
%start iteration
while(1)
    mu = sigmoid(w1, xtrain2);
    % compute gradient
    w1(1)=0;
    g = xtrain2' * (mu - ytrain) + lam * w1;
    % compute hessian
    s = diag(mu .* (1-mu));
    h = xtrain2' * s * xtrain2 + lam * I;
    % update w
    w = w - h\g;
    w1 = w;
    if norm(g,2) < 1e-9
        break;
    end
end

p_y1_test = sigmoid(w,xtest2);
p_y0_test = 1- p_y1_test;
p_y1_train = sigmoid(w,xtrain2);
p_y0_train = 1- p_y1_train;
ytr = zeros(length(xtrain2),1);
yte = zeros(length(xtest2),1);
ytr(find(p_y1_train - p_y0_train >=0),:) = 1;
yte(find(p_y1_test - p_y0_test >=0),:) = 1;
err_train = sum(abs(ytr - ytrain))/length(xtrain2);
err_test = sum(abs(yte - ytest))/length(xtest2);
fprintf('error_rate_test=%f\n',err_test);
fprintf('error_rate_train=%f\n',err_train);
end

%% Sigmoid Function
function u = sigmoid(w, x)
    u = 1 ./ (1 + exp(-w' * x'))';
end