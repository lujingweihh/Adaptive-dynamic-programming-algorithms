clear; close all; clc;

load training_data/state_data.mat;
load training_results/actor_critic.mat


[Kopt, Popt] = dlqr(A,B,Q,R);

x = x0;
x_net = x;
xx = x;
xx_net = x_net;
uu_opt = [];
uu_net = [];

Jreal = 0;

Fsamples = 50;
h = waitbar(0,'Please wait');
for k = 1:Fsamples
    uopt = -Kopt*x;
    x = A*x + B*(uopt);
    xx = [xx x];
    u_net = sim(actor,x_net);
    Jreal = Jreal + x_net'*Q*x_net + u_net'*R*u_net;
    x_net = A*x_net + B*u_net;
    xx_net = [xx_net x_net];
    uu_opt = [uu_opt uopt];
    uu_net = [uu_net u_net];
    waitbar(k/Fsamples,h,['Running...',num2str(k/Fsamples*100),'%']);
end
close(h)

Jopt = x0'*Popt*x0
Jnet = critic(x0)
Jreal

figure,
plot(0:Fsamples,xx,'b-',0:Fsamples,xx_net,'r--','linewidth',1)
xlabel('Time steps');
ylabel('States'); 
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;
figure,
plot(0:Fsamples-1,uu_opt,'b-',0:Fsamples-1,uu_net,'r--','linewidth',1)
legend('Optimal ','NN','Interpreter','latex'); 
xlabel('Time steps');
ylabel('Control');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;




