%% load MNIST dataset
img_tr = loadMNISTImages('train-images.idx3-ubyte');
img_te = loadMNISTImages('t10k-images.idx3-ubyte');
ytrain = loadMNISTLabels('train-labels.idx1-ubyte');
ytest = loadMNISTLabels('t10k-labels.idx1-ubyte');

%%
tr_mean = mean(img_tr, 2);              % compute mean of the training set.
new_img_tr = img_tr - tr_mean;          % substract mean for all training images
new_img_te = img_te - tr_mean; 
% compute covaraince matrix of the training images.
co_tr = 1/60000.*(new_img_tr * new_img_tr');       
[v, d] = eig(co_tr);                    % compute eigenvalues and eigenvectors
e_v = diag(d);
[new_d, idx] = sort(e_v, 'descend');
v10 = v(:,idx(1:10));
% visualize eigenvectors with 10 largest eigenvalues 
for i = 1:10
    h = figure;
    img = reshape(v10(:, i), [28,28]);
    imshow(img);
    name = [num2str(i), '_eigen_b.png'];
    %imwrite(img, name);
end

%%
idx_tr = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20];
idx_te = [4, 3, 2, 19, 5, 9, 12, 1, 62, 8];
v30 = v(:, idx(1:30));          % pick eigenvectors     
plot_tr = new_img_tr(:, idx_tr);    % pick the train images we need
plot_te = new_img_te(:, idx_te);    % pick the test images we need
lam = plot_te' * v30;           % compute lambda (eigenvalues)
xp = v30 * lam';                % compute reconstructed data point
figure;
for i = 1:10
    subplot(2, 10, i)
    img = reshape(plot_tr(:, i), [28,28]);
    imshow(img)
%     name = [num2str(i), 'train_b.png'];
%     imwrite(img, name);       % save images - uncomment since it's only
%     use to save images for report.
end
for i = 1:10
    subplot(2, 10, i+10)
    img = reshape(xp(:, i), [28,28]);
    imshow(img)
%     name = [num2str(i), 'test_reconstruct_b.png'];
%     imwrite(img, name);
end
sgtitle('image reconstruction')
