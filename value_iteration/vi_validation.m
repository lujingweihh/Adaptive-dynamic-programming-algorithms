function vi_validation
%-------------- validate training results --------------
clear; close all; clc;

% information of system & cost function
global A; global B; global Q; global R;

load training_data/state_data.mat
load training_results/actor_critic.mat

[Kopt, Popt] = dlqr(A,B,Q,R);

Fsamples = 50; 

x0 = [1;-1];
x = x0;
x_net = x0;
xx = x.*ones(length(x0),Fsamples+1);
xx_net = x_net.*ones(length(x0),Fsamples+1);
uu_opt = zeros(control_dim,Fsamples);
uu_net = zeros(control_dim,Fsamples);

Jreal = 0;

for k = 1:Fsamples
    u_opt = -Kopt*x;
    x = controlled_system(x,u_opt);
    uu_opt(:,k) = u_opt;
    xx(:,k+1) = x;
    
    u_net = actor(x_net);
    Jreal = Jreal + x_net'*Q*x_net + u_net'*R*u_net;
    x_net = controlled_system(x_net,u_net);
    uu_net(:,k) = u_net;
    xx_net(:,k+1) = x_net;
end

Jopt = x0'*Popt*x0
Jnet = critic(x0)
Jreal

figure(1)
subplot(2,1,1)
plot(0:Fsamples,xx(1,:),'r',0:Fsamples,xx_net(1,:),'b--','linewidth',1)
legend('Optimal','Action network');
ylabel('$x_1$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;
subplot(2,1,2)
plot(0:Fsamples,xx(2,:),'r',0:Fsamples,xx_net(2,:),'b--','linewidth',1)
legend('Optimal','Action network');
xlabel('Time steps');
ylabel('$x_2$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(2)
plot(0:Fsamples-1,uu_opt,'r',0:Fsamples-1,uu_net,'b--','linewidth',1)
legend('Optimal','Action network');
xlabel('Time steps');
ylabel('Controls');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

end

%--------------------------- outpout of system ----------------------------
function y = controlled_system(x,u)
% system matrices
global A; global B;
y = A*x + B*u;  % dot product should be adopt in nolinear systems
end

    



