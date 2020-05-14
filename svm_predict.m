function [accuracy, predict_label]=svm_predict(Xte, Yte, allw, allb)
% allw is nclass * dim
% allb is nclass * 1
% Yte must range from 1 to nclass, other label values will make mistakes
score = Xte * allw'+repmat(allb',[size(Yte,1),1]);
[bb,  c]=sort(score,2,'descend');
predict_label=c(:,1);
temp = predict_label((predict_label-Yte)==0);
right=size( temp,1 );
accuracy=right/size(Yte,1);