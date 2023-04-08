% Code for paper
% J. Lu, Q. Wei, and F.-Y. Wang, "Parallel control for optimal tracking via
% adaptive dynamic programming," IEEE/CAA Journal of Automatica Sinica,
% vol. 7, no. 6, pp. 1662-1674, Nov. 2020.
%
% The paper is available at:
% https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2020.1003426
%
% The main contribution of this paper is to propose an online optimal
% tracking control method for nonaffine systems without reconstructing the
% nonaffine system into an affine form at a certain operating point
%
% P.S. This code is slightly different from the paper 
%
% By J. Lu
% Date: Apr. 8, 2023

%-------------------------------- start -----------------------------------
clear;close all;clc;

global Ga;
global Q; global R; global Qa; global Ra;
global w;
global a1; global a2;
global ra;
global ea;
global v;

% parameters
r = [0.5;0.125]; % reference signal
ur = 1.1947448; % reference control
% this code constructs the augmented system and reference signal first and derives the augmented tracking error system later 
% the paper obtains the tracking error system first and derives the augmented tracking error system later 
% both methods are acceptable
ra = [r;ur]; % augmented reference signal

Ga = [zeros(2,1); diag([1])];

Q = 1*eye(2); R = 1*eye(1);
Qa = diag([diag(Q)' diag(R)']); Ra = 1*diag([1]);


Fsamples = 20000;
T = 0.05; 
ss = T/10; 

x00 = [0;0];
u00 = 0;
w00 = zeros(12,1); % initial critic network weights
x = [x00;u00;0;0;w00];
w = x(6:end);
x0 = x;
xx = [x];
uu = [x(3)];
rr = [r];
vv = [];

ww = [w];

% learning rates are different from the paper due to the introduction of the probing noise
a1 = 20;
a2 = 0.1;

Jo = 0;
Ja = 0;

rpb_k = Fsamples - 200;


h = waitbar(0,'please wait');
for k = 1:Fsamples
      
    if k~= 1 && norm(ea(1:3),2) <= 0.5 && k <= rpb_k % probing noise
        x = [ra + 0.5*(rand(3,1)-0.5);x(4:end)'];
    end
    
    tspan = 0:ss:T;
    [t,x]= ode45(@pc_ot_adp_ode, tspan, x);
        
    Jo = Jo + x(length(t),4);
    Ja = Ja + x(length(t),5);
    
    x = [ x(length(t),1:3),0,0,x(length(t),6:end) ];
    w = x(6:end);
    
    rr = [rr r];
    xx = [xx x'];
    uu = [uu x(3)];
    vv = [vv v];
    ww = [ww w'];
    
    waitbar(k/Fsamples,h,['Running...',num2str(k/Fsamples*100),'%']);
end
close(h);

Jo
Ja

% result
figure(1), % states 
plot(T*((1:size(rr,2))-1),rr(1,:),'r','linewidth',1),hold on;
plot(T*((1:size(xx,2))-1),xx(1,:),'b--','linewidth',1);
xlabel('Time (s)');
ylabel('$x_1$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(2), % states 
plot(T*((1:size(rr,2))-1),rr(2,:),'r','linewidth',1),hold on;
plot(T*((1:size(xx,2))-1),xx(2,:),'b--','linewidth',1);
xlabel('Time (s)');
ylabel('$x_2$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(3), % real control
plot(T*((1:size(uu,2))-1),uu,'b','linewidth',1)
xlabel('Time (s)');
ylabel('$u$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(4), % virtual control
plot(T*((1:size(vv,2))-1),vv,'b','linewidth',1)
xlabel('Time (s)');
ylabel('$v$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(5), % wc
plot(T*((1:size(ww,2))-1),ww,'linewidth',1)
xlabel('Time (s)');
ylabel('Critic newtork weights');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;




