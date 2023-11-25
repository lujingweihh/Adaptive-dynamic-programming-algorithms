% J. Lu, X. Wang, Q. Wei, F.-Y. Wang. Nearly optimal stabilization of unknown continuous-time nonlinear systems: A new parallel control approach.
% Code for Case I in Example 2
% By J. Lu

%-------------------------------- start -----------------------------------
clear; close all; clc;

global M; global g; global L;
global k1; global k2; global k3;
global Ga;
global Q; global R; global Ra;
global Ta;
global dw;
global wa;

Ta = 1;
Ga = [zeros(2,1); (1/Ta)*diag([1])];

M = 0.3; L = 0.6; g = 10; k1 = 0.8; k2 = 0.2; k3 = 1; 

Q = 5*eye(2);R = 0.1*eye(1);
Qa = diag([diag(Q)' diag(R)']); Ra = 1*diag([1]);

Fsamples = 100000;
T = 0.1; 
ss = T/10; 

% state, control, and critic and action NN weights
x00 = 1*[1;0];
wc = [1;0;0;1;0;1;zeros(6,1)]; 

x = [x00;-1;0;0;wc]; 
wa = x(6:end);
x0 = x;
xx = [x];
uu = [];

phi = eg2_c1_nn_pf(x(1:3));
pphi = [];
pp = [];

% NN learning law
dw = zeros(length(phi),1);


wwa = [wa];
wwc = [x(6:end)];

ac = 100; 
aa = 0.08;

cw = 1e-2;

rpb_k = Fsamples - 200; % remove the probing noise at the last 200 steps

Gamma = eye(length(phi));

h = waitbar(0,'please wait');
for k = 1:Fsamples
      
    % probing noise
    if norm(x(1:2),2) <= 0.2 && k <= rpb_k
        x = [x(1);x(2);x(3)+2*(rand(1,1)-0.5);0;0;x(6:end)'];
        phi = eg2_c1_nn_pf(x(1:3));
    end
    
    tspan = 0:ss:T;
    [t,x]= ode45(@eg2_c1_ode, tspan, x);
    
    phi_next = eg2_c1_nn_pf(x(length(t),1:3));
    pphi = phi_next-phi;
    phi = phi_next;
    
    pp = x(length(t),4);
    
    % critic learning law
    dw = - ac*pphi/((1+pphi'*pphi)^2)*(pp+pphi'*x(length(t),6:end)');
   
    % action learning law
    if norm(dw,2) <= cw
        wa = wa + (-aa*Gamma*(wa - x(length(t),6:end)'));
        phi = eg2_c1_nn_pf(x(length(t),1:3));
    end
    
    x = [x(length(t),1:3),0,0,x(length(t),6:end)];
    xx = [xx x'];
    uu = [uu x(3)];
    wwa = [wwa wa];
    wwc = [wwc x(6:end)'];
    
    waitbar(k/Fsamples,h,['Running...',num2str(k/Fsamples*100),'%']);
end
close(h);

wc = xx(6:end,end)'
wa = wa'

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



