clear;close all;
load('linear_svm.mat');
%% train stage for primal problem
[num,dim]=size(X_train);
C = 1;
[w_primal,b_primal] = CVX_prim(X_train,labels_train,C);
out = labels_train.*(X_train*w_primal+b_primal);
out = round(out*100)/100;
ind = find( out==1 );
figure
gscatter(X_train(:,1),X_train(:,2),labels_train);
hold on
x = min(X_train(:)):0.01:max(X_train(:)); y = (-x*w_primal(2)-b_primal)/w_primal(1);
plot(y,x);
hold on
for i = 1:length(ind)
    gscatter(X_train(ind(i),1),X_train(ind(i),2),'support vector','g');
end
legend('1','-1','Decision Function','Support Vector','Location','Best');
title('train stage for SVM primal problem');
%% test stage for primal problem
[num_,dim_]=size(X_test);
labels_predict = zeros(num_,1);
X = X_test*w_primal+b_primal;
for i = 1:num_
    if X(i) >= 0
        labels_predict(i) = 1;
    else
        labels_predict(i) = -1;
    end
end
error = labels_predict - labels_test;
error_ind = find(error ~= 0);
disp(['Test error:',num2str(length(error_ind))]);
disp(['Accuracy:',num2str((num_-length(error_ind))/num_)]);
figure 
gscatter(X_test(:,1),X_test(:,2),labels_test);
hold on
x = min(X_test(:)):0.01:max(X_test(:)); y = (-x*w_primal(2)-b_primal)/w_primal(1);
plot(y,x);
hold on
legend('1','-1','Decision Function','Location','Best');
title('test stage for SVM primal problem');
disp(['w: ',num2str(w_primal(1)),' , ',num2str(w_primal(2))]);
disp(['Bias: ',num2str(b_primal)]);

%% train stage for dual problem
[w_dual,b_dual] = CVX_dual(X_train,labels_train,C);
out_ = labels_train.*(X_train*w_dual+b_dual);
out_ = round(out_*100)/100;
ind_ = find( out_==1 );
figure
gscatter(X_train(:,1),X_train(:,2),labels_train);
hold on
x = min(X_train(:)):0.01:max(X_train(:)); y = (-x*w_dual(2)-b_dual)/w_dual(1);
plot(y,x);
hold on
for i = 1:length(ind_)
    gscatter(X_train(ind_(i),1),X_train(ind_(i),2),'support vector','g');
end
legend('1','-1','Decision Function','Support Vector','Location','Best');
title('train stage for SVM dual problem');
%% test stage for dual problem
%[num_,dim_]=size(X_test);
labels_predict_ = zeros(num_,1);
X = X_test*w_dual+b_dual;
for i = 1:num_
    if X(i) >= 0
        labels_predict_(i) = 1;
    else
        labels_predict_(i) = -1;
    end
end
error = labels_predict_ - labels_test;
error_ind_ = find(error ~= 0);
disp(['Test error:',num2str(length(error_ind_))]);
disp(['Accuracy:',num2str((num_-length(error_ind_))/num_)]);
figure 
gscatter(X_test(:,1),X_test(:,2),labels_test);
hold on
x = min(X_test(:)):0.01:max(X_test(:)); y = (-x*w_dual(2)-b_dual)/w_dual(1);
plot(y,x);
hold on
legend('1','-1','Decision Function','Location','Best');
title('test stage for SVM dual problem');
disp(['w: ',num2str(w_dual(1)),' , ',num2str(w_dual(2))]);
disp(['Bias: ',num2str(b_dual)]);