% J. Lu, X. Wang, Q. Wei, F.-Y. Wang. Nearly optimal stabilization of unknown continuous-time nonlinear systems: A new parallel control approach.
% Code for Case II in Example 1
% By J. Lu

%-------------------------------- start -----------------------------------
clear; close all; clc;

global A; global B;
global Ba;
global Qa; global Ra;
global Ta;
global dw;
global wa;

A = [0 1;-9 -3]; B = [0; 1];

Ta = 1;
Aa = [A B; zeros(1,2) -1/Ta];
Ba = [zeros(2,1); (1/Ta)*diag([1])];

Q = 1*eye(2); R = 1*eye(1);
Qa = diag([diag(Q)' diag(R)']); Ra = 1*diag([1]);

[Kaopt, Paopt] = lqr(Aa,Ba,Qa,Ra);

Fsamples = 10000;
T = 0.1; 
ss = T/10; 

% state, control, and critic and action NN weights
x00 = [1;-1];
x = [x00;0;0;1;0;0;1;0;1];
wa = x(5:end);
x0 = x;
xx = [x];
uu = [];

phi = eg1_c2_nn_pf(x(1:3));
pphi = [];
pp = [];

% NN learning law
dw = zeros(length(phi),1);

wwa = [wa];
wwc = [x(5:end)];

ac = 30;
aa = 0.1;

cw = 1e-2;

rpb_k = Fsamples - 100; % remove the probing noise at the last 100 steps

Gamma = eye(length(phi));

h = waitbar(0,'please wait');
for k = 1:Fsamples
    
    % probing noise
    if norm(x(1:2),2) <= 1 && k <= rpb_k
        x = [x(1);x(2);x(3)+10*(rand(1,1)-0.5);0;x(5:end)'];
        phi = eg1_c2_nn_pf(x(1:3));
    end
    
    tspan = 0:ss:T;
    [t,x]= ode45(@eg1_c2_ode, tspan, x);
    
    phi_next = eg1_c2_nn_pf(x(length(t),1:3));
    pphi = phi_next-phi;
    phi = phi_next;
    
    pp = x(length(t),4);
    
    % critic learning law
    dw = - ac*pphi/((1+pphi'*pphi)^2)*(pp+pphi'*x(length(t),5:end)');
   
    % action learning law
    if norm(dw,2) <= cw
        wa = wa + (-aa*Gamma*(wa - x(length(t),5:end)'));
        phi = eg1_c2_nn_pf(x(length(t),1:3));
    end
    
    x = [x(length(t),1:3),0,x(length(t),5:end)];
    xx = [xx x'];
    uu = [uu x(3)];
    wwa = [wwa wa];
    wwc = [wwc x(5:end)'];
    
    waitbar(k/Fsamples,h,['Running...',num2str(k/Fsamples*100),'%']);
end
close(h);

wc = xx(5:end,end)'
wa = wa'
wopt = [Paopt(1,1); 2*Paopt(1,2); 2*Paopt(1,3); Paopt(2,2); 2*Paopt(2,3); Paopt(3,3)]'
difference_initial = x0(5:end)' - wopt
difference_final = wc - wopt

% result
figure(1), % states 
subplot(2,1,1)
plot(T*((1:size(xx,2))-1),xx(1,:),'linewidth',1)
xlabel('Time (s)');
ylabel('$x_1$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;
subplot(2,1,2)
plot(T*((1:size(xx,2))-1),xx(2,:),'linewidth',1)
xlabel('Time (s)');
ylabel('$x_2$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(2), % control
plot(T*((1:size(uu,2))-1),uu(1,:),'linewidth',1)
xlabel('Time (s)');
ylabel('$u$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(3), % wc
plot(T*((1:size(xx,2))-1),xx(5:end,:),'linewidth',1)
xlabel('Time (s)');
ylabel('Critic newtork weights');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(4), % wa
plot(T*((1:size(wwa,2))-1),wwa,'linewidth',1)
xlabel('Time (s)');
ylabel('Action newtork weights');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;



