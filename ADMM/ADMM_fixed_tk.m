%Assignment 3, question c
%ADMM for solving Lasso Problem 
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

%from the constraint
ys = xs;

%initialize x ,y ,and lamda
xk = zeros(n,1);
yk = zeros(n,1);
lamda = zeros(n,1);

%Given value of tau
tau = 1;

%initialize fixed step size tx
tk = 0.0005;

%ADMM terminates when norm(xk-xs)/norm(xs) smaller than given epsi = 10^-3
%denote norm(xk-xs)/norm(xs) = cr1 (criterion 1)
%norm(xk - yk) = cr2 (criterion 2)
%set k as the counter
epsi = 10^-3;
cr1 = 1;
cr2 = 1;
fxopt = 1/2*(norm(A*xs-b))^2+tau*norm(ys,1);
k = 1;

%two stopping criteria
while cr1 >= epsi | cr2 >= 0.1
    %directly solve x subproblem (see PDF document for details) 
    xk = (A'*A+tk*eye(n))\(A'*b + tk*yk +lamda);
    
    %solve y sub problem using proximal mapping method 
    %define u (as explained in PDF) 
    u = xk - lamda/tk;
    
    for i = 1:length(u)
        if u(i) >= tau/tk
            yk(i) = u(i) - tau/tk;
        elseif u(i) <= -tau/tk
            yk(i) = u(i) + tau/tk;
        else 
            yk(i) = 0;
        end
    end
    
    %update lamda 
    lamda = lamda - tk*(xk - yk);
    
    %update criterion 
    cr1(k) = norm(xk-xs)/norm(xs);
    cr2(k) = norm(xk-yk);
    
    %update k
    k = k+1;
    
    %update error 
    fx(k) = 1/2*(norm(A*xk-b))^2+tau*norm(yk,1);
    error(k) = abs(fx(k) - fxopt);
end 

%tic/toc used to report the CPU time
toc 

%k as a counter 
k

%plot
figure 
plot(1:k-1,cr1)
title('Fixed step ADMM satisfied the termination criteria using epsilon=10^-3');
xlabel('Number of iteration');
ylabel('Termination Criteria');

figure
plot(log(1:k),log(error))
title('Log-log plot of function value error of fixed step ADMM using epsilon=10^-3');
xlabel('log of number of iteration');
ylabel('log of error');
