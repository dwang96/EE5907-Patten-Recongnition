%% load MNIST dataset
tic
img_tr = loadMNISTImages('train-images.idx3-ubyte');
img_te = loadMNISTImages('t10k-images.idx3-ubyte');
ytrain = loadMNISTLabels('train-labels.idx1-ubyte');
ytest = loadMNISTLabels('t10k-labels.idx1-ubyte');

%% convert ytrain and ytest to indicator matrix
ytrain_ind = zeros(60000, 10);
ytest_ind = zeros(10000, 10);
idx_0 = find(ytrain == 0);
idx_1 = find(ytrain == 1);
idx_2 = find(ytrain == 2);
idx_3 = find(ytrain == 3);
idx_4 = find(ytrain == 4);
idx_5 = find(ytrain == 5);
idx_6 = find(ytrain == 6);
idx_7 = find(ytrain == 7);
idx_8 = find(ytrain == 8);
idx_9 = find(ytrain == 9);
newind = diag(ones(1, 10));
ytrain_ind(idx_0, :) = repmat(newind(1,:), length(idx_0), 1);
ytrain_ind(idx_1, :) = repmat(newind(2,:), length(idx_1), 1);
ytrain_ind(idx_2, :) = repmat(newind(3,:), length(idx_2), 1);
ytrain_ind(idx_3, :) = repmat(newind(4,:), length(idx_3), 1);
ytrain_ind(idx_4, :) = repmat(newind(5,:), length(idx_4), 1);
ytrain_ind(idx_5, :) = repmat(newind(6,:), length(idx_5), 1);
ytrain_ind(idx_6, :) = repmat(newind(7,:), length(idx_6), 1);
ytrain_ind(idx_7, :) = repmat(newind(8,:), length(idx_7), 1);
ytrain_ind(idx_8, :) = repmat(newind(9,:), length(idx_8), 1);
ytrain_ind(idx_9, :) = repmat(newind(10,:), length(idx_9), 1);

idx_0 = find(ytest == 0);
idx_1 = find(ytest == 1);
idx_2 = find(ytest == 2);
idx_3 = find(ytest == 3);
idx_4 = find(ytest == 4);
idx_5 = find(ytest == 5);
idx_6 = find(ytest == 6);
idx_7 = find(ytest == 7);
idx_8 = find(ytest == 8);
idx_9 = find(ytest == 9);
newind = diag(ones(1, 10));
ytest_ind(idx_0, :) = repmat(newind(1,:), length(idx_0), 1);
ytest_ind(idx_1, :) = repmat(newind(2,:), length(idx_1), 1);
ytest_ind(idx_2, :) = repmat(newind(3,:), length(idx_2), 1);
ytest_ind(idx_3, :) = repmat(newind(4,:), length(idx_3), 1);
ytest_ind(idx_4, :) = repmat(newind(5,:), length(idx_4), 1);
ytest_ind(idx_5, :) = repmat(newind(6,:), length(idx_5), 1);
ytest_ind(idx_6, :) = repmat(newind(7,:), length(idx_6), 1);
ytest_ind(idx_7, :) = repmat(newind(8,:), length(idx_7), 1);
ytest_ind(idx_8, :) = repmat(newind(9,:), length(idx_8), 1);
ytest_ind(idx_9, :) = repmat(newind(10,:), length(idx_9), 1);

%% Linear regression
tr_mean = mean(img_tr, 2);              % compute mean of the training set.
new_img_tr = img_tr - tr_mean;          % substract mean for all training images
co_tr = 1/60000.*(new_img_tr * new_img_tr');       
[v, d] = eig(co_tr);                    % compute eigenvalues and eigenvectors
e_v = diag(d);
[new_d, idx] = sort(e_v, 'descend');
vnew = v(:, idx);                       % find the eigenvectors corresponding with 30 largest eigenvalues
v30 = vnew(:, 1:30);
eigendigit_tr = img_tr' * v30;          % 30 * 60000
eigendigit_te = img_te' * v30;          % 30 * 10000
% introduce bias term
bias_tr = ones(60000, 1);
bias_te = ones(10000, 1);
eigendigit_train = [bias_tr, eigendigit_tr];
eigendigit_test = [bias_te, eigendigit_te];
w = inv(eigendigit_train' * eigendigit_train) * eigendigit_train' * ytrain_ind;
label_test = eigendigit_test * w;
[~, pseudo_label] = max(label_test,[], 2);
real_test = pseudo_label - 1;
test_err = abs(real_test - ytest);
test_err(test_err ~= 0) = 1;
acc_te = 1 - sum(test_err)/10000;
toc

%% polynomial regression
eigenpoly_tr = [eigendigit_train, eigendigit_tr.^2];
eigenpoly_te = [eigendigit_test, eigendigit_te.^2];
wpoly = inv(eigenpoly_tr' * eigenpoly_tr) * eigenpoly_tr' * ytrain_ind;
label_test_poly = eigenpoly_te * wpoly;
[~, pseudo_label_poly] = max(label_test_poly,[], 2);
real_test_poly = pseudo_label_poly - 1;
test_err = abs(real_test_poly - ytest);
test_err(test_err ~= 0) = 1;
acc_te_poly = 1 - sum(test_err)/10000;