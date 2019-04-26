%Assignment 3, question b
%FISTA with backtracking line search for solving Lasso Problem 
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

%FISTA terminates when norm(xk-xs)/norm(xs) smaller than given epsi = 10^-3
%denote norm(xk-xs)/norm(xs) = criterion (cr)
%set j as the counter
epsi = 10^-3;
cr = norm(xk-xs)/norm(xs);
fxopt = 1/2*(norm(A*xs-b))^2+tau*norm(xs,1);

%define x1 = x(k-1), x2 = x(k-2)
x1 = xk;
x2 = xk;
k = 1;

while cr >= epsi
    %Initialize the fixed step size tk
    tk = 0.1;
    
    y = x1 + (k-2)/(k+1)*(x1-x2);
    
    %for Ax-b not 0, the subgradient of g(y) is the gradient of g(y)
    gg = A'*(A*y-b);
    
    %implement backtracking line search for ISTA
    %denote RHS = RHS of line search inequality
    gx = 1;
    RHS = 0;
    
    while gx > RHS
        %xknew refers to the updated xk 
        u = y-tk*gg;
        
        %select element from u to determine xknew
        for i = 1:n
            if u(i) >= tk*tau
                xknew(i,1) = u(i) - tk*tau;
            elseif u(i) <= -tk*tau
                xknew(i,1) = u(i) + tk*tau;
            else 
                xknew(i,1) = 0;
            end
        end
        gx = 1/2*(norm(A*xknew-b))^2;
        gy = 1/2*(norm(A*y-b))^2;
        RHS = gy + gg'*(xknew-y) + 1/2/tk*(norm(xknew-y))^2;
        beta = 0.5;
        tk = beta*tk;
    end 
    %update the xk 
    x2 = x1;
    x1 = xknew;
    xk = xknew;

    %update the gradient of f
    gg = A'*(A*xk-b);
    fx(k) = 1/2*(norm(A*xk-b))^2+tau*norm(xk,1);
    error(k) = abs(fx(k) - fxopt);
    k = k+1; 
    
    %store the value of Euclidean norm
    cr(k) = norm(xk-xs)/norm(xs);
    
%     plot(j,enorm(j),'-*')
%     hold on
%     drawnow
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
title('FISTA with line search satisfied the termination criteria using epsilon=10^-3');
xlabel('Number of iteration');
ylabel('Termination Criteria');
figure
plot(log(2:k),log(error))
title('Log-log plot of function value error of FISTA with line search using epsilon=10^-3');
xlabel('log of number of iteration');
ylabel('log of error');