%-------------------------- obtain the initial admissible control ----------------------------
clear; close all; clc;

load training_data/state_data.mat

[K, P] = dlqr(A,B,Q,100*R);

actor_target = -K*x_train;

cover = 1;
if isempty(dir('training_data/actor_init.mat')) == 1 || cover == 1
    % action network
    actor_init_middle_num = 15;
    actor_init_epoch = 10000;
    actor_init_err_goal = 1e-9;
    actor_init_lr = 0.01;
    actor_init = newff(minmax(x_train), [actor_init_middle_num control_dim], {'tansig' 'purelin'},'trainlm');
    actor_init.trainParam.epochs = actor_init_epoch;
    actor_init.trainParam.goal = actor_init_err_goal;
    actor_init.trainParam.show = 10;
    actor_init.trainParam.lr = actor_init_lr;
    actor_init.biasConnect = [1;0];
    
    actor_init = train(actor_init, x_train, actor_target);
    
    save training_data/actor_init actor_init
else
    load training_data/actor_init
end


%-------------------------- test the initial control ----------------------------
x = x0;
x_net = x;

xx = x;
xx_net = x_net;
uu = [];
uu_net = [];

Fsamples = 200;
JK = 0;
Jnet = 0;

h = waitbar(0,'Please wait');
for k = 1:Fsamples
    u = -K*x;
    u_net = actor_init(x_net);
    JK = JK + x'*Q*x + u'*R*u;
    Jnet = Jnet + x_net'*Q*x_net + u_net'*R*u_net;
    x = A*x + B*u;
    xx = [xx x];
    x_net = A*x_net + B*u_net;
    xx_net = [xx_net x_net];
    uu = [uu u];
    uu_net = [uu_net u_net];
    waitbar(k/Fsamples,h,['Running...',num2str(k/Fsamples*100),'%']);
end
close(h)

JK
Jnet

figure,
plot(0:Fsamples,xx,'b-',0:Fsamples,xx_net,'r--','linewidth',1)
xlabel('Time steps');
ylabel('States'); 
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure,
plot(0:Fsamples-1,uu,'b-',0:Fsamples-1,uu_net,'r--','linewidth',1)
xlabel('Time steps');
ylabel('Control');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;



