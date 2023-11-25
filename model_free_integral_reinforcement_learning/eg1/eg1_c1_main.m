% J. Lu, X. Wang, Q. Wei, F.-Y. Wang. Nearly optimal stabilization of unknown continuous-time nonlinear systems: A new parallel control approach.
% Code for Case I in Example 1
% By J. Lu

%-------------------------------- start -----------------------------------
clear; close all; clc;

global A; global B; 
global Aa; global Ba; 
global Q; global R;
global Qa; global Ra;
global Ta;
global Kopt; global Kaopt;
global u; 
global ua;


A = [0 1;-9 -3]; B = [0; 1];

Ta = 1; 
Aa = [A B; zeros(1,2) -1/Ta]; Ba = [zeros(2,1); (1/Ta)*diag([1])];

Q = 1*eye(2); R = 1*eye(1);
Qa = diag([diag(Q)' diag(R)']); Ra = 1e0*diag([1]);

[Kopt, Popt] = lqr(A,B,Q,R);
[Kaopt, Paopt] = lqr(Aa,Ba,Qa,Ra);

Fsamples = 400; 
T = 0.05; 
ss = T/10; 

% origin system
% state and control
x0 = [1;1];
x = [x0;0];
xx = [x(1:2)];
uu = [-Kopt*x0];

% performance
J = 0;

h = waitbar(0,'please wait');
for k = 1:Fsamples
    tspan = 0:ss:T;
    [t,x]= ode45(@eg1_c1_ode, tspan, x);
    J = J + x(length(t),3);
    x = [x(length(t),1:2),0];
    xx = [xx x(1:2)'];
    uu = [uu u];

    waitbar(k/Fsamples,h,['Running...',num2str(k/Fsamples*100),'%']);
end
close(h);

J

% aas
% state and control
xa = [x0;uu(:,1);0];
xxa = [xa(1:2)];
uua = [uu(:,1)];

% performance
Ja = 0;

h = waitbar(0,'please wait');
for k = 1:Fsamples
    tspan = 0:ss:T;
    [t,xa]= ode45(@eg1_c1_parallel_ode, tspan, xa);
    Ja = Ja + xa(length(t),4);
    xa = [xa(length(t),1:3),0];
    xxa = [xxa xa(1:2)'];
    uua = [uua xa(3)];
    
    waitbar(k/Fsamples,h,['Running...',num2str(k/Fsamples*100),'%']);
end
close(h);

Ja

Jopt = x0'*Popt*x0

fprintf('Ta: %d; Ra: %d; difference: %d.\n',Ta,Ra,Ja-Jopt)


% result
figure(1), % states 
subplot(2,1,1)
plot(T*((1:size(xx,2))-1),xx(1,:),'r',T*((1:size(xxa,2))-1),xxa(1,:),'b--','linewidth',1)
xlabel('Time (s)');
ylabel('$x_1$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;
subplot(2,1,2)
plot(T*((1:size(xx,2))-1),xx(2,:),'r',T*((1:size(xxa,2))-1),xxa(2,:),'b--','linewidth',1)
xlabel('Time (s)');
ylabel('$x_2$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(2), % control
plot(T*((1:size(uu,2))-1),uu(1,:),'r',T*((1:size(uua,2))-1),uua(1,:),'b--','linewidth',1)
xlabel('Time (s)');
ylabel('$u$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;



