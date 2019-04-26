%Assignment 3, question a
%ISTA with fixed step size for solving Lasso Problem 
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

%initialize vector of x = 0
xk = zeros(n,1);

%Given value of tau
tau = 1;

%for Ax-b not 0, the subgradient of g(x) is the gradient of g(x)
gg = A'*(A*xk-b);

%ISTA terminates when norm(xk-xs)/norm(xs) smaller than given epsi = 10^-3
%denote norm(xk-xs)/norm(xs) = criterion (cr)
%set j as the counter
epsi = 10^-3;
cr = norm(xk-xs)/norm(xs);
j = 0;
fxopt = 1/2*(norm(A*xs-b))^2+tau*norm(xs,1);

while cr >= epsi 
    %Initialize the fixed step size tk
    tk = 0.0005;
    
    %xknew refers to the updated xk 
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
    cr(j+1) = norm(xk-xs)/norm(xs);
    
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
plot(1:j+1,cr)
title('Fixed step size ISTA satisfied the termination criteria using epsilon=10^-3');
xlabel('Number of iteration');
ylabel('Termination Criteria');
figure
plot(log(1:j),log(error))
title('Log-log plot of function value error of fixed step size ISTA using epsilon=10^-3');
xlabel('log of number of iteration');
ylabel('log of error');