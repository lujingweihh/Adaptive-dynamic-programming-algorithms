function vi_algorithm
% This demo checks the feasibility of value iteration adaptive dynamic programming 

% By J. Lu
% Date: May 6, 2020 

%-------------------------------- start -----------------------------------
clear; close all; clc;

% information of system & cost function
global A; global B; global Q; global R;

load training_data/state_data.mat

% action network
actor_middle_num = 8;
actor_epoch = 20000;
actor_err_goal = 1e-7;
actor_lr = 0.05;
actor = newff(minmax(x_train), [actor_middle_num control_dim], {'tansig' 'purelin'},'trainlm');
actor.trainParam.epochs = actor_epoch; % max epochs
actor.trainParam.goal = actor_err_goal; % tolerance error
actor.trainParam.show = 10;  % interval
actor.trainParam.lr = actor_lr; % learning rate - traingd,traingdm
actor.biasConnect = [1;0];% bias 

% critic network
critic_middle_num = 8;
critic_epoch = 10000;
critic_err_goal = 1e-7;
critic_lr = 0.01;
critic = newff(minmax(x_train), [critic_middle_num 1], {'tansig' 'purelin'},'trainlm');
critic.trainParam.epochs = critic_epoch;
critic.trainParam.goal = critic_err_goal; 
critic.trainParam.show = 10;  
critic.trainParam.lr = critic_lr; 
critic.biasConnect = [1;0];
critic_last = critic;

epoch = 50;
tol = 1e-9;
performance_index = zeros(1,epoch + 1);

critic_set = cell(1,epoch);
actor_set = cell(1,epoch);

figure(1),plot(0,0,'*')
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;
hold on;
h = waitbar(0,'please wait');
for i = 1:epoch
    % update action network
    actor_target = zeros(1,size(x_train,2));
    if i ~= 1
        for j = 1:size(x_train,2)
            x = x_train(:,j);
            if x == zeros(state_dim,1)
                ut = zeros(control_dim,1);
            else
                objective = @(u) cost_function(x,u) + critic(controlled_system(x,u));
                u0 = actor(x);
                ut = fminunc(objective,u0);
                actor_target(j) = ut;
            end
        end
    end

    actor = train(actor, x_train, actor_target);    

    % update critic network
    if i == 1
        critic_target = cost_function(x_train,0);
    else
        x_next = controlled_system(x_train,actor(x_train));
        critic_target = cost_function(x_train,(actor(x_train))) + critic_last(x_next);
    end
    for j = 1:size(zeros_index,2)
        critic_target(:,zeros_index(j)) = zeros(1,1);
    end
    critic = train(critic,[x_train,-x_train],[critic_target,critic_target]);
    
    if i ~= 1 && mse(critic(x_train),critic_last(x_train)) <= tol 
        critic_set{i} = critic;
        actor_set{i} = actor;
        break;
    end
    
    critic_last = critic;
    
    critic_set{i} = critic;
    actor_set{i} = actor;
    
    performance_index(i+1) = critic(x0);
    figure(1),plot(i,performance_index(i+1),'*')
    waitbar(i/epoch,h,['Training controller...',num2str(i/epoch*100),'%']);
end
close(h);

figure(1),
xlabel('Iterations');
ylabel('Iterative $V(x_0)$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;
hold off;

save training_results/actor_critic critic_set actor_set critic actor

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






