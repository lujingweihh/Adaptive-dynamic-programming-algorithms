% This demo checks the feasibility of the integral reinforcement learning (IRL) algorithm

% By J. Lu
% Date: Nov. 23, 2020 

%-------------------------------- start -----------------------------------
clear; close all; clc;

% parameters
global K; global u; global A; global B; global Q; global R;
P = zeros(4,4);
uu = 0; % record control signal u

% system matrices
A = [-0.0665   11.5       0         0;
         0     -2.5      2.5        0;
       -9.5      0    -13.736  -13.736;
        0.6      0       0         0];
B = [0 0 13.736 0]';
Q = diag([1,1,1,1]);
R = 1;

% initial condition
x0 = [0 0.1 0 0 0]; % x(5) is the integral of x'Qx + u'Ru

% a stable K
K = [0.8267    1.7003    0.7049    0.4142];
pole = eig(A - B*K)

% simulation parameters
Fsamples = 120;
T = 0.05; % sample time
SS = 0.001; % step size for simulation
j = 0;
nop = 10;
tol = 1e-4;

figure(1),hold on;
xlabel('Time (s)');
ylabel('State trajectories'); 
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;
for k=1:Fsamples
    j = j + 1;
    tspan = 0:SS:T;
    [t,x]= ode23('irl_ode',tspan,x0);
    uu = [uu u];
    figure(1),plot(t + T*(k-1),x(:,1),'b',t + T*(k-1),x(:,2),'r',t + T*(k-1),x(:,3),'g',t + T*(k-1),x(:,4),'k','linewidth',1),hold on;
    x1 = x(length(t),1:4);
    X(j,:) = [ x0(1)^2 x0(1)*x0(2) x0(1)*x0(3) x0(1)*x0(4) ...
               x0(2)^2 x0(2)*x0(3) x0(2)*x0(4)...
               x0(3)^2 x0(3)*x0(4) x0(4)^2 ] - ...
             [ x1(1)^2 x1(1)*x1(2) x1(1)*x1(3) x1(1)*x1(4) ...
               x1(2)^2 x1(2)*x1(3) x1(2)*x1(4)...
               x1(3)^2 x1(3)*x1(4) x1(4)^2 ];
    
    Y(j,:) = x(length(t),5) - x(1,5);
    x0 = [x(length(t),1:4),x(length(t),5)];

    if mod(k,nop) == 0 && norm(X) > tol
        weights = pinv(X)*Y;
        P=[ weights(1)    weights(2)/2  weights(3)/2  weights(4)/2;...
            weights(2)/2  weights(5)    weights(6)/2  weights(7)/2;...
            weights(3)/2  weights(6)/2  weights(8)    weights(9)/2;...
            weights(4)/2  weights(7)/2  weights(9)/2  weights(10)      ];
        K = inv(R)*B'*P;
        jj = 1;
        X = zeros(nop,10);
        Y = zeros(nop,1);
        j = 0;
    end
end

P
[~,Popt] = lqr(A,B,Q,R)

% plot result
figure(1), hold off;

figure(2),plot(T*[0:Fsamples],uu,'b','linewidth',1)
xlabel('Time (s)');
ylabel('Control trajectory'); 
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;


