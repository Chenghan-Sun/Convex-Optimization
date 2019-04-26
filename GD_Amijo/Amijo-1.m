%Assignmnet 1, question 4
clear all, clc

%Gradient method using Amijo line search 
tic
%input data points 
m = 500;
n = 1000;
A1 = randn(n,m);
b = sign(rand(m,1)-0.5);

%catch ai from matrix A1 and define vector bi to construct A2 for gradient
%calculation w.r.t w and c 
A2 = zeros(n,m);
for i = 1:m
    ai = A1(:,i);
    bi = b(i,1);
    A2(:,i) = bi*ai;
end 
A2 = A2';

%initialize c and vector w
c = 1;
w = ones(n,1);

%initial guess of vector xk
xki = [w;c];

%calculate gradient of w and c and put as gradient of f as a vector
one = ones(m,1);
p = one./(one+exp(-A2*w-b.*c));
gdc = -1/m*b'*(one-p);
gdw = -1/m*A2'*(one-p);
gdf = [gdw;gdc];

%amijo parameters
alpha = 0.1;
beta = 0.7;

%gradient method terminate when L-2 norm of gradient of f smaller than
%given value 10^-4 10^-3 or 10^-2
epsi = 10^-4;
enorm = 1;
j = 0;

while enorm > epsi
    %initialize the RHS of Amijo condition 
    %fx2 refers to initial value of f(x-tk*gdf)
    %initialize tk
    rhs = 0;
    fx2 = 1;
    tk = 1;
    
    %construct a loop for using Amijo condition 
    while fx2 > rhs
        
        %fx1 refers to f(x)
        %construct a loop to update fx1 and fx2
        %xknew refers to the updated xk using gradient method
        fx1 = 0;
        fx2 = 0;
        xknew = xki-tk*gdf;
        for i = 1:m
            ai = A1(:,i);
            ai1 = [ai;1];
            fx1 = fx1 + 1/m * log(1+exp(-b(i)*(xki'*ai1)));
            fx2 = fx2 + 1/m * log(1+exp(-b(i)*(xknew'*ai1)));
        end
        rhs = fx1 - alpha*tk*(norm(gdf))^2;
        
        %update the tk
        tk = beta*tk;
    end
    %update the xk 
    xki = xknew;
    
    %store the value of Euclidean norm
    enorm(j+1) = norm(gdf);

    %update the gradient of f
    w = xki(1:1000,1);
    c = xki(1001,1);
    p = one./(one+exp(-A2*w-b.*c));
    gdc = -1/m*b'*(one-p);
    gdw = -1/m*A2'*(one-p);
    gdf = [gdw;gdc];
    
    %j as a counter for number of iterations 
    j = j+1;
end 

%tic/toc used to report the CPU time
toc 

%j as a counter 
j

%plot Euclidean norm of the gradient in the course of the algorithm
plot(1:j,enorm)
title('Euclidean norm of the gradient in the course of the algorithm for epsilon = 10^-4');
xlabel('Number of iteration');
ylabel('Euclidean norm');

