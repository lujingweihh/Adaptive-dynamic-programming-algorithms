function pi_algorithm
% This demo checks the feasibility of the policy iteration adaptive dynamic 
% programming  algorithm

% By J. Lu
% Date: Apr. 21, 2020 

%-------------------------------- start -----------------------------------
clear; close all; clc;

% information of system & cost function
global A; global B; global Q; global R;

load training_data/state_data.mat;
load training_data/actor_init.mat;

% action network
actor = actor_init;

actor_epoch = 20000;
actor_err_goal = 1e-9;
actor_lr = 0.01;
actor.trainParam.epochs = actor_epoch; 
actor.trainParam.goal = actor_err_goal; 
actor.trainParam.show = 10; 
actor.trainParam.lr = actor_lr; 

% critic network
critic_middle_num = 15;
critic_epoch = 10000;
critic_err_goal = 1e-9;
critic_lr = 0.01;
critic = newff(minmax(x_train), [critic_middle_num 1], {'tansig' 'purelin'},'trainlm');
critic.trainParam.epochs = critic_epoch;
critic.trainParam.goal = critic_err_goal; 
critic.trainParam.show = 10;  
critic.trainParam.lr = critic_lr; 
critic.biasConnect = [1;0];

epoch = 10;
eval_step = 400;
performance_index = ones(1,epoch + 1);

figure(1),hold on;
h = waitbar(0,'Please wait');
for i = 1:epoch
    % update critic
    % evaluate policy
    critic_target = evaluate_policy(actor, x_train, eval_step);
    critic = train(critic,x_train,critic_target); 
    
    performance_index(i) = critic(x0);
    figure(1),plot(i,performance_index(i),'*'),xlim([1 epoch]),hold on;
    
    waitbar(i/epoch,h,['Training controller...',num2str(i/epoch*100),'%']);
    if i == epoch
        break;
    end
    
    % update actor
    actor_target = zeros(control_dim,size(x_train,2));
    for j = 1:size(x_train,2)
        x = x_train(:,j);
        if x == zeros(state_dim,1)
            ud = zeros(control_dim,1);
        else
            objective = @(u) cost_function(x,u) + critic(controlled_system(x,u));
            u0 = actor(x);
            ud = fminunc(objective, u0);
        end
        actor_target(:,j) = ud;
    end
    
    actor = train(actor, x_train, actor_target);
end
close(h)

figure(1),
xlabel('Iterations');
ylabel('$V(x_0)$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;
hold off;

save training_results/actor_critic actor critic
end

%---------------------------- evaluate policy -----------------------------
function y = evaluate_policy(actor,x,eval_step)
critic_target = zeros(1,size(x,2));
for k = 1:eval_step
    uep = actor(x);
    critic_target = critic_target +  cost_function(x,uep);
    x = controlled_system(x,uep);
end
y = critic_target;
end
%--------------------------- outpout of system ----------------------------
function y = controlled_system(x,u)
% system matrices
global A; global B;
y = A*x + B*u;  % dot product should be adopt in nolinear systems
end
%----------------------------- cost function ------------------------------
function y = cost_function(x,u)
global Q; global R;
y = (diag(x'*Q*x) + diag(u'*R*u))';
end

