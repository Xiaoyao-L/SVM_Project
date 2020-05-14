function [w,b] = sgd_svm(x,y,C)

[n,d] = size(x);
epoch = 100000;
w =zeros(d,1);
b = sum(y-x*w)/n;
losses = [];
time = [];

 for i = 1:epoch
     tic;
     tem = [x,y]; tem = tem(randperm(size(tem,1)),:);
     x = tem(:,1:d);y = tem(:,d+1); step = 0.01;
     loss = [];   
     for k = 1:n
         yk = y(k,1); xk = x(k,:);
         e_ = yk*(xk*w+b);
         if(e_<1)
             w = w + (step/n)*yk*xk'; b = b + (step/n)*yk;
             loss = [loss;1-e_];
         else
             w = w; b = b;
             loss = [loss;0];
         end
     end
     target = w'*w/2+C*sum(loss);
     losses = [losses;target];
     toc;
     time = [time;toc];
     
 end
 
 index = find(losses == min(losses));
 disp(['objective value is ',num2str(min(losses))]);
 disp(['number of iterations is ',num2str(index)]);
 disp(['CPU time: ',num2str(sum(time(1:index)))]);
 
 
 figure();
 plot((1:length(losses)),losses,'-ro');
 legend('Loss after each epoch');


end


