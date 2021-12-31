function result1 = test(test_data, model)
    test_data = test_data.';
    test_data(:, model.drop_cond) = [];
    test_data = normalize(test_data, 1);
    knn = model.knn;
    net = model.net;
    tree = model.tree;
    bayes = model.bayes;
    
    %Predict with centroid of kmeans
    [~,idx_test] = pdist2(model.C,test_data,'euclidean','Smallest',1);
    idx_test = double(idx_test(:) == model.cluster);
    voting_group = [categorical(idx_test),...
        categorical(knn.predict(test_data)), ...
        categorical(classify(net, test_data)), ...
        categorical(tree.predict(test_data)), ...
        categorical(bayes.predict(test_data))];
    
    result1 = mode(voting_group, 2);
end