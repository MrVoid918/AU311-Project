labeled_X = load("Data1/data1.mat").data1.';
labeled_Y = double(load("Data1/label1.mat").label1.');
%labeled_Y = load("Data1/label1.mat").label1.';
unlabeled_X = load("Data1/data1_unlabel.mat").data1_unlabel.';

%Drop Features if is all-zeros
drop_cond = all(~labeled_X,1) & all(~unlabeled_X, 1);
labeled_X( :, drop_cond ) = [];
unlabeled_X( :, drop_cond ) = [];
%}

concat_X = [labeled_X ;unlabeled_X];
concat_X = normalize(concat_X, 1);
normed_lab_X = concat_X(1:100, :);
normed_unlab_X = concat_X(101:1000, :);

%Baseline 0.5
opts = statset('Display', 'final');
[idx, C] = kmeans(normed_lab_X, 2, 'Replicates', 10);%, 'Options', opts);
idx = double(idx(:) == 2);
c_mat = confusionmat(labeled_Y, idx);
acc = sum(diag(c_mat)) / sum(c_mat, 'all');

%label with graph-based method
graphMd_1 = fitsemigraph(normed_lab_X, labeled_Y, normed_unlab_X);
graphMd_1.FittedLabels;

%label with self-training method
selftr_1 = fitsemiself(normed_lab_X, labeled_Y, normed_unlab_X, ...
    'Learner', templateEnsemble('AdaBoostM1', 100, 'tree'));
selftr_1.FittedLabels;

rng(2)  %Reproducibility
[idx, C] = kmeans([normed_lab_X; normed_unlab_X], 2, 'Replicates', 20, 'Options', opts);
%[idx, C] = kmeans(normed_unlab_X, 2, 'Replicates', 20, 'Options', opts);
%With Unlabeled Only
idx = double(idx(:) == 2);
%Do note that label swap may occur and cause low accuracy
%[~,idx_test] = pdist2(C,normed_lab_X,'euclidean','Smallest',1);
%c_mat = confusionmat(labeled_Y, idx(1:100));
%acc = sum(diag(c_mat)) / sum(c_mat, 'all');
%idx_test = double(idx_test(:) == 1);
c_mat = confusionmat(labeled_Y, idx(1:100));
acc = sum(diag(c_mat)) / sum(c_mat, 'all');

knn = fitcknn(normed_lab_X, labeled_Y);
knn_pred = knn.predict(normed_unlab_X);
c_mat = confusionmat(knn_pred, idx(101:1000));
acc = sum(diag(c_mat)) / sum(c_mat, 'all');

layers = [
    featureInputLayer(size(normed_lab_X, 2),"Name","featureinput")
    fullyConnectedLayer(256,"Name","fc_1")
    reluLayer("Name","relu_1")
    fullyConnectedLayer(128,"Name","fc_2")
    reluLayer("Name","relu_2")
    fullyConnectedLayer(64,"Name","fc_3")
    reluLayer("Name","relu_3")
    fullyConnectedLayer(2,"Name","fc_4")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classification")];

options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false);

net = trainNetwork(labeled_X, categorical(labeled_Y), layers, options);
net_pred = classify(net, normed_unlab_X);



%{
%Determine whether classes are balanced
%{
[C,ia,ic] = unique(labeled_Y);
a_counts = accumarray(ic,1);
value_counts = [C, a_counts]
%}


 %{
layers = [
    featureInputLayer(size(normed_lab_1, 2),"Name","featureinput")
    fullyConnectedLayer(256,"Name","fc_1")
    reluLayer("Name","relu_1")
    fullyConnectedLayer(128,"Name","fc_2")
    reluLayer("Name","relu_2")
    fullyConnectedLayer(64,"Name","fc_3")
    softmaxLayer("Name","sigmoid")
    classificationLayer("Name","classification")];

options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false);

net = trainNetwork(labeled_X, labeled_Y, layers, options);

%}
%}
%}

function pred_ = logistic_regression(X_, w_)
    pred_ = 1./(1 + exp(-X_ * w_));
end

function loss_ = loss_function(X_, w_, Y_)  %We choose different class
    assert(size(Y_, 2) == 1, "Last Dimension should be 1");
    loss_ = sum(Y_ .* (X_ * w_) - log(1 + exp(X_ * w_)));
end

function grad_ = loss_grad(X_, w_, Y_)
    grad_ =  X_.' * (Y_ - logistic_regression(X_, w_));
end