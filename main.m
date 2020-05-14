clear all;close all;
load('linear_svm.mat');
Xtr = X_train;
Xte = X_test;
Ytr = labels_train;
Yte = labels_test;
C = 1;
%% SGD
[w,b] = sgd_svm(Xtr,Ytr,C);
%%
y=Xtr*w+b;
y(y>0)=1;
y(y<=0)=-1;
result2=[Ytr,y];
error=size(find(Ytr.*y<0),1);
error_train=error/size(Ytr,1);
figure 
gscatter(Xtr(:,1),Xtr(:,2),labels_train);
hold on
x = min(Xtr(:)):0.01:max(Xtr(:)); y = (-x*w(2)-b)/w(1);
plot(y,x);
hold on
legend('1','-1','Decision Function','Location','Best');
title('Using SGD in SVM primal problem(train stage)');
%% test stage
y=Xte*w+b;
y(y>0)=1;
y(y<=0)=-1;
result1=[Yte,y];
error=size(find(Yte.*y<0),1);
error_test=error/size(Yte,1);
disp(['Test error: ',num2str(error)]);
disp(['Accuracy: ',num2str((size(Yte,1)-error)/size(Yte,1))]);
figure 
gscatter(X_test(:,1),X_test(:,2),labels_test);
hold on
x = min(X_test(:)):0.01:max(X_test(:)); y = (-x*w(2)-b)/w(1);
plot(y,x);
hold on
legend('1','-1','Decision Function','Location','Best');
title('Using SGD in SVM primal problem(test stage)');

disp(['w: ',num2str(w(1)),' , ',num2str(w(2))]);
disp(['Bias: ',num2str(b)]);

