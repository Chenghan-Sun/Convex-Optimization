%Assignmnet 4, question b
%CGD method with randomized rule and fixed step size for solving Lasso Problem 
clear all, clc

tic
%input data points, xs is the true solution that we want to find
m = 300;
n = 500;
s = 2;
A = randn(m,n);
xs = zeros(n,1);
picks = randperm(n);
xs(picks(1:s)) = 100*randn(s,1);
b = A*xs;

%initialize x 
xk = zeros(n,1);

%Given value of tau
tau = 1;

%initialize gradient vector
gd = zeros(n,1);

%initialize fixed CGD step size alpha
alpha = 0.001;

%CGD method with randomized rule terminates when norm(xk-xs)/norm(xs) smaller than given epsi = 10^-3
%denote norm(xk-xs)/norm(xs) = cr (criterion)
%set k as the counter
epsi = 10^-3;
cr = 1;
fxopt = 1/2*(norm(A*xs-b))^2+tau*norm(xs,1);
k = 1;

%Initialize randomized ik 
ik = randi([1,n]);

%Apply stopping criterion 
while cr(k) >= epsi
    
    %update the gradient
    u = A*xk-b;
    A1 = A';
    gd1 = A1(ik,:)*u;
    gd2 = sign(xk(ik))*tau; 
    gd(ik) = gd1+gd2;
    
    %update xk
    xk(ik) = xk(ik)-alpha*gd(ik);
    
    %update criterion 
    cr(k+1) = norm(xk-xs)/norm(xs);
    
    %update k
    k = k+1;
    
    %update error 
    fx(k) = 1/2*(norm(A*xk-b))^2+tau*norm(xk,1);
    error(k) = abs(fx(k) - fxopt);
    
    %update ik
    ik = randi([1,n]);
end 

%tic/toc used to report the CPU time
toc 

%report tau
tau

%k as a counter 
k

%plot
figure 
plot(1:k,cr)
title('Randomized CGD method satisfied the termination criteria using epsilon=10^-3');
xlabel('Number of iteration');
ylabel('Termination Criteria');

figure
plot(log(1:k),log(error))
title('Log-log plot of function value error of Randomized CGD method using epsilon=10^-3');
xlabel('log of number of iteration');
ylabel('log of error');
