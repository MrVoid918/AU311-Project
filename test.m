function result1 = test1(test_data, model)
    test_data(:, model.drop_labels) = [];
    test_data = normalize(test_data, 1);
end

function result2 = test2(test_data, model)
    test_data(:, model.drop_labels) = [];
    test_data = normalize(test_data, 1);
end

function result3 = test3(test_data, model)
    test_data(:, model.drop_labels) = [];
    test_data = normalize(test_data, 1);
end