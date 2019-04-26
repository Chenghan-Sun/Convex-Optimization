%Assignment 2, question 4
clear all, clc

%Proximal gradient method for solving Lasso Problem 
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
tau = 0.1;

%for Ax-b not 0, the subgradient of g(x) is the gradient of g(x)
gg = A'*(A*xk-b);

%Proximal gradient method terminates when norm(xk-xs)/norm(xs) smaller than
%given epsi = 10^-2 10^-4 10^-6
%denote norm(xk-xs)/norm(xs) = enorm
%set j as the counter
epsi = 10^-6;
enorm = norm(xk-xs)/norm(xs);
j = 0;
fxopt = 1/2*(norm(A*xs-b))^2+tau*norm(xs,1);
while enorm >= epsi
    %Initialize the fixed step size tk
    tk = 0.001;
    
    %xknew refers to the updated xk using proximal gradient method
    u = xk-tk*gg;
    
    %select element from vector u to determine xknew
    for i = 1:n
        if u(i) >= tk*tau
            xknew(i) = u(i) - tk*tau;
        elseif u(i) <= -tk*tau
            xknew(i) = u(i) + tk*tau;
        else 
            xknew(i) = 0;
        end
    end
 
    %update the xk 
    xk = xknew';
    
    %update the gradient of f
    gg = A'*(A*xk-b);
    fx(j+1) = 1/2*(norm(A*xk-b))^2+tau*norm(xk,1);
    error(j+1) = abs(fx(j+1) - fxopt);
    j = j+1; 
    
    %store the value of Euclidean norm
    enorm(j+1) = norm(xk-xs)/norm(xs);
    
    if j > 2
        if enorm(j)-enorm(j-1)>=0
            tau = tau*0.5;
        end
    end 
%     plot(j,enorm(j),'-*')
%     hold on
%     drawnow
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
title('Proximal gradient Method satisfied the terminate criteria using epsilon=10^-6');
xlabel('Number of iteration');
ylabel('Terminate Criteria');
figure
loglog(1:j,error)
title('Log-log plot of function value error of Proximal gradient Method using epsilon=10^-6');
xlabel('log of Number of iteration');
ylabel('log of error');