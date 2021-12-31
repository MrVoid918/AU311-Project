%{
test_data1 = load('D:\博三上\模式识别导论\project\评分\Data\data1_test.mat').test_data;
test_label1 = load('D:\博三上\模式识别导论\project\评分\Data\label1_test.mat').test_label';

test_data2 = load('D:\博三上\模式识别导论\project\评分\Data\data2_test.mat').test_data;
test_label2 = load('D:\博三上\模式识别导论\project\评分\Data\label2_test.mat').test_label';

test_data3 = load('D:\博三上\模式识别导论\project\评分\Data\data3_test.mat').test_data;
test_label3 = load('D:\博三上\模式识别导论\project\评分\Data\label3_test.mat').test_label';
%}
% 加载完成后，test_data: 360*3000, test_label: 3000*1
sample_data = rand(360, 3000);
sample_label = categorical(rand(3000, 1) > .5);

%加载模型 
%model = load("model/model.mat");
model = load("model.mat");
pred1 = test(sample_data, model);
assert(isequal(size(pred1), size(sample_label)));

%若需要训练
%label_data, label, unlabel_data = dataloader(...);
%model = train(label_data, label, unlabel_data);

% 测试
%{
pred1 = test1(test_data1, model);
pred2 = test2(test_data2, model);
pred3 = test3(test_data3, model);%}
% every dataset has 3000 test data
acc1 = length(find(pred1 == test_label1))/3000;
fprintf('acc1: %4.2f\n', acc1*100);

acc2 = length(find(pred2 == test_label2))/3000;
fprintf('acc2: %4.2f\n', acc2*100);

acc3 = length(find(pred3 == test_label3))/3000;
fprintf('acc3: %4.2f\n', acc3*100);
%}