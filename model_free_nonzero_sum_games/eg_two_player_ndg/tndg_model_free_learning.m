function tndg_model_free_learning
% ''Parallel Control for Nonzero-Sum Games With Completely Unknown
%   Nonlinear Dynamics via Reinforcement Learning''
%
% Code for 2-player nonlinear differential game (TNDG)
%
% By J. Lu
% Date: Aug. 12, 2024

%-------------------------------- start -----------------------------------
clear; close all; clc;

global Ga1; global Ga2;
global Q1; global R11; global R12;
global Q2; global R21; global R22;
global Qa1; global Ra11; global Ra12;
global Qa2; global Ra21; global Ra22;
global w1; global w2;
global dw1; global dw2;

% system information & performance indices
n = 2;
m1 = 1;
m2 = 1;
eta = n + m1 + m2;

Ga1 = [ zeros(n,m1); eye(m1); zeros(m2,m1) ]; % $\mathcal{G}_1$
Ga2 = [ zeros(n,m2); zeros(m1,m2); eye(m2) ]; % $\mathcal{G}_2$

Q1 = 2*eye(n); R11 = 2*eye(m1); R12 = 2*eye(m2);
Q2 = 1*eye(n); R21 = 1*eye(m1); R22 = 1*eye(m2);
Qa1 = [ Q1,zeros(n,m1+m2);zeros(m1,n),R11,zeros(m1,m2);zeros(m2,n+m1),R12 ]; Ra11 = 1*eye(m1); Ra12 = 0*eye(m2);
Qa2 = [ Q2,zeros(n,m1+m2);zeros(m1,n),R21,zeros(m1,m2);zeros(m2,n+m1),R22 ]; Ra21 = 0*eye(m1); Ra22 = 1*eye(m2);

Fsamples = 50000;
T = 0.2; 
ss = T/10; 

w1 = [  1   0   1   1   ...
        1   1   1   ... 
        1  -1   ... 
        1 ]'; 
w2 = w1;

x_ini = 1;
u_ini = 1;

x0 = x_ini*[ 1; -1 ];
u0 = u_ini*[ 0; 0 ];
x = [ x0; u0; 0; 0; 0; 0; w1; w2 ];

nn_length = length(w1);

xx = [ x ];
uu1 = [];
uu2 = [];

ell = length(w1); % number of hidden neurons
L = ell; %  need L >= ell to satisfy rank($\mathcal{D}_i$) = $\ell_i$

phi = tndg_nn_pf(x(1:eta));
pphi = zeros(ell,L+1); %  number of historical data (L) + real-time data (1)
pp1 = zeros(L+1,1);
pp2 = zeros(L+1,1);

dw1 = zeros(nn_length,1);
dw2 = zeros(nn_length,1);

ww1 = [ w1 ];
ww2 = [ w2 ];

l1 = 0.5; % learning rates
l2 = 0.5;

l1_final = 1e-5;
l2_final = 1e-5;

l_decay_interval = 1000;

l1_decay = (l1-l1_final)/(0.8*Fsamples/l_decay_interval);
l2_decay = (l2-l2_final)/(0.8*Fsamples/l_decay_interval);

i = 1;

h = waitbar(0,'Please wait');
for k = 1:Fsamples
    if norm(x(1:n),2) > 1e3 % the system is unstable and need to restart the learning
        break
    end

    if mod(k,l_decay_interval) == 0
        l1 = l1 - l1_decay;
        l2 = l2 - l2_decay;
    end

    if l1 < l1_final
        l1 = l1_final;
    end
    
    if l2 < l2_final
        l2 = l2_final;
    end

    if (norm(x(1:n),2) <= 0.1)  % reset controls
        u0 = 2*u_ini*[ (rand(1)-0.5); (rand(1)-0.5) ];
        x = [ x(1:n)'; u0; 0; 0; 0; 0; x(eta+5:end)' ];
        phi = tndg_nn_pf(x(1:eta));
    end

    tspan = 0:ss:T;
    [t,x]= ode45(@tndg_ode, tspan, x);
    
    phi_next = tndg_nn_pf(x(length(t),1:eta));

    if i <= L + 1
        pphi(:,i) = phi_next-phi;
        pp1(i) = x(length(t),eta+3);
        pp2(i) = x(length(t),eta+4);

        phi = phi_next;

        i = i + 1;
    else
        for j = 1:L
            pphi(:,j) = pphi(:,j+1);
            pp1(j) = pp1(j+1);
            pp2(j) = pp2(j+1);
        end
        pphi(:,L+1) = phi_next-phi;
        pp1(L+1) = x(length(t),eta+3);
        pp2(L+1) = x(length(t),eta+4);

        phi = phi_next;
    end

    w1 = x(length(t),eta+5:eta+4+nn_length)';
    w2 = x(length(t),eta+5+nn_length:eta+4+2*nn_length)';   
    
    if i >= L + 1
        dw1 = zeros(nn_length,1);
        dw2 = zeros(nn_length,1);
        for j = 1:L+1
            dw1 = dw1 - l1*pphi(:,j)/((1+pphi(:,j)'*pphi(:,j))^2)*(pp1(j)+pphi(:,j)'*w1);
            dw2 = dw2 - l2*pphi(:,j)/((1+pphi(:,j)'*pphi(:,j))^2)*(pp2(j)+pphi(:,j)'*w2);
        end
    end    

    x = [x(length(t),1:eta),0,0,0,0,x(length(t),eta+5:end)];

    xx = [ xx x' ];
    uu1 = [ uu1 x(n+1) ];
    uu2 = [ uu2 x(n+2) ];
    ww1 = [ ww1 w1 ];
    ww2 = [ ww2 w2 ];
    
    waitbar(k/Fsamples,h,['Running...',num2str(k/Fsamples*100),'%']);
end
close(h);

w1_display = w1'
w2_display = w2'

% result
figure(1), % states
subplot(2,1,1)
plot(T*((1:size(xx,2))-1),xx(1,:),'linewidth',1)
xlabel('Time (s)');
ylabel('$x_{1}$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;
subplot(2,1,2)
plot(T*((1:size(xx,2))-1),xx(2,:),'linewidth',1)
xlabel('Time (s)');
ylabel('$x_{2}$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(2), % control 1
plot(T*((1:size(uu1,2))-1),uu1,'linewidth',1)
xlabel('Time (s)');
ylabel('$u_1$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(3), % control 2
plot(T*((1:size(uu2,2))-1),uu2,'linewidth',1)
xlabel('Time (s)');
ylabel('$u_2$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(4), % w1
plot(T*((1:size(ww1,2))-1),ww1,'linewidth',1)
xlabel('Time (s)');
ylabel('$\mathcal{W}_{1}$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(5), % w2
plot(T*((1:size(ww2,2))-1),ww2,'linewidth',1)
xlabel('Time (s)');
ylabel('$\mathcal{W}_{2}$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

end
%---------------------------- tndg dynamics -------------------------------
function xdot = tndg_ode(~,x)
global Ga1; global Ga2;
global Q1; global R11; global R12;
global Q2; global R21; global R22;
global Qa1; global Ra11; global Ra12;
global Qa2; global Ra21; global Ra22;
global w1; global w2;
global dw1; global dw2;

x_state = [ x(1); x(2); ];

u1 = x(3);
u2 = x(4);

u = [ u1; u2 ];

s = [ x_state; u ];

dphi = tndg_nn_pf_dphi(s);
    
v1 = -0.5*inv(Ra11)*Ga1'*dphi'*w1;
v2 = -0.5*inv(Ra22)*Ga2'*dphi'*w2;

g1 = [ 0; cos(2*x(1))+2 ];
g2 = [ 0; sin(4*x(1)^2)+2 ];  

tndg_dynamics = [  -2*x(1) + x(2);...
                   -0.5*x(1) - x(2) + x(1)^2*x(2) + 0.05*x(2)*(cos(2*x(1))+2)^2 + 0.05*x(2)*(sin(4*x(1))+2)^2  ]...
                + g1*u1 + g2*u2; 

xdot = [ tndg_dynamics;...
         v1;...
         v2;...
         x_state'*Q1*x_state + u1'*R11*u1 + u2'*R12*u2;...
         x_state'*Q2*x_state + u1'*R21*u1 + u2'*R22*u2;...
         s'*Qa1*s + v1'*Ra11*v1 + v2'*Ra12*v2;...
         s'*Qa2*s + v1'*Ra21*v1 + v2'*Ra22*v2;...
         dw1;...
         dw2 ];
end
%------------------- activation function vector of critic ------------------
function y = tndg_nn_pf(x)
x1 = x(1);
x2 = x(2);
x3 = x(3);
x4 = x(4);

y = [ x1^2; x1*x2; x1*x3; x1*x4;...
      x2^2; x2*x3; x2*x4;...
      x3^2; x3*x4;...
      x4^2 ];
end
%-------------- derivative of critic activation function vector -----------
function dphi = tndg_nn_pf_dphi(x)
x1 = x(1);
x2 = x(2);
x3 = x(3);
x4 = x(4);

dphi = [
         [  2*x1,          0,          0,          0  ];
         [    x2,         x1,          0,          0  ]; 
         [    x3,          0,         x1,          0  ]; 
         [    x4,          0,          0,         x1  ]; 
         [     0,       2*x2,          0,          0  ];
         [     0,         x3,         x2,          0  ];
         [     0,         x4,          0,         x2  ];
         [     0,          0,       2*x3,          0  ];
         [     0,          0,         x4,         x3  ];
         [     0,          0,          0,       2*x4  ];   
       ];
end




