labeled_X = load("Data1/data1.mat").data1.';
labeled_Y = double(load("Data1/label1.mat").label1.');
%labeled_Y = load("Data1/label1.mat").label1.';
unlabeled_X = load("Data1/data1_unlabel.mat").data1_unlabel.';

%Drop Features if is all-zeros
drop_cond = all(~labeled_X,1) & all(~unlabeled_X, 1);
labeled_X( :, drop_cond ) = [];
unlabeled_X( :, drop_cond ) = [];

concat_X = [labeled_X ;unlabeled_X];
concat_X = normalize(concat_X, 1);
normed_lab_X = concat_X(1:100, :);
normed_unlab_X = concat_X(101:1000, :);

rng(3)
train_test_split = randperm(100);
NX_train = normed_lab_X(train_test_split(1:70), :);
Y_train = labeled_Y(train_test_split(1:70), :);
NX_test = normed_lab_X(train_test_split(70:100), :);
Y_test = labeled_Y(train_test_split(70:100), :);

cv = cvpartition(length(Y_train), 'KFold', 7);


knn = fitcknn(NX_train, Y_train);
knn_pred = knn.predict(NX_test);
c_mat = confusionmat(knn_pred, Y_test)
acc = sum(diag(c_mat)) / sum(c_mat, 'all')

%{
%label with graph-based method
graphMd_1 = fitsemigraph(normed_lab_X, labeled_Y, normed_unlab_X);
graphMd_1.FittedLabels;

%label with self-training method
selftr_1 = fitsemiself(normed_lab_X, labeled_Y, normed_unlab_X, ...
    'Learner', templateEnsemble('AdaBoostM1', 100, 'tree'));
selftr_1.FittedLabels;
%}

rng(3)  %Reproducibility
opts = statset('Display', 'final');
[idx, C] = kmeans([NX_train; normed_unlab_X], 2, 'Replicates', 20, 'Options', opts);
idx = double(idx(:) == 2);
cluster = 1;
%Do note that label swap may occur and cause low accuracy
[~,idx_test] = pdist2(C,NX_test,'euclidean','Smallest',1);
%c_mat = confusionmat(labeled_Y, idx(1:100));
%acc = sum(diag(c_mat)) / sum(c_mat, 'all');
idx_test = double(idx_test(:) == cluster);
c_mat = confusionmat(Y_test, idx_test)
acc = sum(diag(c_mat)) / sum(c_mat, 'all')



layers = [
    featureInputLayer(size(normed_lab_X, 2),"Name","featureinput")
    fullyConnectedLayer(256,"Name","fc_1")
    dropoutLayer(0.3, "Name", "dropout_1")
    reluLayer("Name","relu_1")
    fullyConnectedLayer(128,"Name","fc_2")
    dropoutLayer(0.3, "Name", "dropout_2")
    reluLayer("Name","relu_2")
    fullyConnectedLayer(64,"Name","fc_3")
    reluLayer("Name","relu_3")
    fullyConnectedLayer(2,"Name","fc_4")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classification")];

options = trainingOptions('adam', ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false);

net = trainNetwork(NX_train, categorical(Y_train), layers, options);
net_pred = classify(net, NX_test);
c_mat = confusionmat(net_pred, categorical(Y_test))
acc = sum(diag(c_mat)) / sum(c_mat, 'all')

tree = fitctree(NX_train, categorical(Y_train));
tree_pred = tree.predict(NX_test);
c_mat = confusionmat(tree_pred, categorical(Y_test))
acc = sum(diag(c_mat)) / sum(c_mat, 'all')


bayes = fitcnb(NX_train, categorical(Y_train), 'Distribution', 'kernel');
bayes_pred = bayes.predict(NX_test);
c_mat = confusionmat(bayes_pred, categorical(Y_test))
acc = sum(diag(c_mat)) / sum(c_mat, 'all')

save model1.mat drop_cond C cluster knn net tree bayes
%{
voting_group = [categorical(graphMd_1.FittedLabels),...
    categorical(selftr_1.FittedLabels), ...
    categorical(idx(101:1000)), categorical(knn_pred),...
    net_pred, tree_pred, bayes_pred];

Voting_result = mode(voting_group, 2);
%}



%Determine whether classes are balanced
%{
[C,ia,ic] = unique(labeled_Y);
a_counts = accumarray(ic,1);
value_counts = [C, a_counts]
%}
%}