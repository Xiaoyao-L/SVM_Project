function [w,b] = CVX_prim(x,labels,C)
X = x;
[num,dim]=size(X);
cvx_begin
    variables w(dim) b xi(num);
    minimize (sum(w.^2)/2 + C * sum(xi));
    subject to
        labels.* (X * w+b) >= 1-xi;
        xi >= 0;
cvx_end
end

