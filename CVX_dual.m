
function [w,b] = CVX_dual(x,labels,C)
Y=labels; 
X=x; 
[n,dim] = size(X);
K=X*X';
cvx_begin 
    variables alphas(n) 
    maximize( sum(alphas) -  0.5*quad_form(Y.*alphas,K))
    subject to
       alphas > 0;
       alphas < C;
       sum(alphas.*Y) == 0;
cvx_end
w=X'*(alphas.*Y);
epsilon = 0.0001;
svii = find( alphas > epsilon & alphas < (C - epsilon));
b =  (1/length(svii))*sum(Y(svii)) - (1/length(svii))*sum(X(svii,:))*w;
end