% J. Lu, X. Wang, Q. Wei, F.-Y. Wang. Nearly optimal stabilization of unknown continuous-time nonlinear systems: A new parallel control approach.
% Code for Case II in Example 2
% By J. Lu

%-------------------------------- start -----------------------------------
clear; close all; clc;

global M; global g; global L;
global k1; global k2; global k3;
global Ga;
global Ta;
global wa;

Ta = 1;
Ga = [zeros(2,1); (1/Ta)*diag([1])];

M = 0.3; L = 0.6; g = 10; k1 = 0.8; k2 = 0.2; k3 = 1; 

Fsamples = 150; 
T = 0.05; 
ss = T/10; 

wa = [10.4347996132347;4.74187769922673;4.59458795955741;-0.140646935292767;5.46611496181774;2.82980614212775;0.00266940936499440;0.0224520628430915;0.00480024500336202;2.63761282635573;-0.179632857958409;-0.150728264946674];

x00 = [1;0];
u00 = 2;
x = [x00;u00]; 
xx0 = [x];
uu0 = [u00];

for k = 1:Fsamples
    tspan = 0:ss:T;
    [t,x]= ode45(@eg2_c2_ode, tspan, x);
   
    x = x(length(t),:);
    xx0 = [xx0 x'];
    uu0 = [uu0 x(3)];
end


u01 = -2;
x = [x00;u01]; 
xx1 = [x];
uu1 = [u01];

for k = 1:Fsamples
    tspan = 0:ss:T;
    [t,x]= ode45(@eg2_c2_ode, tspan, x);
   
    x = x(length(t),:);
    xx1 = [xx1 x'];
    uu1 = [uu1 x(3)];
end


figure(1), % 3d
plot3(xx0(1,:),xx0(2,:),uu0,'r','linewidth',1),hold on;
plot3(xx1(1,:),xx1(2,:),uu1,'b-.','linewidth',1),hold on;
plot3(x00(1,:),x00(2,:),u00,'ro','MarkerFaceColor','r','linewidth',1,'MarkerSize',6),hold on;
plot3(x00(1,:),x00(2,:),u01,'bo','MarkerFaceColor','b','linewidth',1,'MarkerSize',6),hold on;
xlim([-0.5 1.1])
xlabel('$x_1$','Interpreter','latex');
ylabel('$x_2$','Interpreter','latex');
zlabel('$u$','Interpreter','latex');
h=legend('$u^1_0=2$','$u^2_0=-2$');
set(h,'Interpreter','latex','Position',[0.154401278625391 0.266349206349207 0.236911728463504 0.106714288938613]);    
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;
view([426.299998823416 14.7740121081114])



