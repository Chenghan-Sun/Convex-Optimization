%Assignment 2, question 4
clear all, clc

%Subgradient method for solving Lasso Problem 
tic
%input data points, xs is the optimal solution that we want to find
m = 100;
n = 500;
s = 5;
A = randn(m,n);
xs = zeros(n,1);
picks = randperm(n);
xs(picks(1:s)) = randn(s,1);
b = A*xs;

%initialize vector of x = 0
xk = zeros(n,1);

%guess the value of tau
tau = 0.0031;

%for Ax-b not 0, the subgradient of gx is the gradient of gx
%sg1 = subgradient of 1/2||Axk-b||^2; sg2 = subgradient of L1 norm of xk
sg1 = A'*(A*xk-b);
sg2 = -tau; %choose g=-1 at x = 0
sg = sg1+sg2;

%subgradient method terminates when norm(xk-xs)/norm(xs) smaller than
%given epsi = 10^-2 10^-4 10^-6
%denote norm(xk-xs)/norm(xs) = enorm
%set j as the counter

epsi = 10^-4;
enorm = norm(xk-xs)/norm(xs);
j = 0;
fxopt = 1/2*(norm(A*xs-b))^2+tau*norm(xs,1);
while enorm >= epsi
    %Initialize the fixed step size tk
    tk = 0.001;
 
    %xknew refers to the updated xk using subgradient method
    xknew = xk-tk*sg;
 
    %update the xk 
    xk = xknew;
    
    %update the gradient of f
    sg1 = A'*(A*xk-b);
    sg2 = sign(xk)*tau; 
    sg = sg1+sg2;
    
    fx(j+1) = 1/2*(norm(A*xk-b))^2+tau*norm(xk,1);
    error(j+1) = abs(fx(j+1) - fxopt);
    
    j = j+1; 
    if j > 2
        if enorm(j)-enorm(j-1)>=0
            tau = tau*0.5;
        end
    end 
    %store the value of Euclidean norm
    enorm(j+1) = norm(xk-xs)/norm(xs);
end 

%tic/toc used to report the CPU time
toc 

%report tau
tau

%j as a counter 
j

%plot 
figure 
plot(0:j,enorm)
title('Subgradient Method satisfied the terminate criteria using epsilon=10^-4');
xlabel('Number of iteration');
ylabel('Terminate Criteria');
figure
loglog(1:j,error)
title('Log-log plot of function value error of Subgradient Method using epsilon=10^-4');
xlabel('log of Number of iteration');
ylabel('log of error');

