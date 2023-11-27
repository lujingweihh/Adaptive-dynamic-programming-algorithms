% This demo checks the feasibility value iteration adaptive dynamic 
% programming using an arbitrary positive semi-definite function as
% the initial value function

% By J. Lu
% Date: May 6, 2020 

%-------------------------------- start -----------------------------------
clear; close all; clc;

% utilize vi_state_data.m to generate training data and then run this file
load state_data

% the value function of linear systems can be approximated using quadratic
% form and eigenvalues of its corresponding matrix are 3.9645 and 11.0355
weights_critic_ini = 10*[1;-0.5;0.5]; 

% eigenvalues can be computed by
% P_ini = [ weights_critic_ini(1) weights_critic_ini(2)/2; weights_critic_ini(2)/2 weights_critic_ini(3) ]
% eig(P_ini)

weights_critic = weights_critic_ini; 
weights_actor = zeros(2,1); 

xV = vi_pf_critic(x_train);
xV_minus = vi_pf_critic(-x_train);
xu = vi_pf_actor(x_train);

epoch = 20;
performance_index = zeros(1,epoch + 1);

performance_index(1) = weights_critic'*vi_pf_critic(x0);

weights_critic_set = weights_critic;
weights_actor_set = [];

figure(1),plot(0,performance_index(1),'*'),hold on;
h = waitbar(0,'please wait');
for i = 1:epoch
    % update action network
    % by optimal tool
    if i ~= 1
        actor_target = zeros(control_dim,size(x_train,2));
        for j = 1:size(x_train,2)
            x = x_train(:,j);
            if x == zeros(state_dim,1)
                ut = zeros(control_dim,1);
            else
                objective = @(u) vi_cost_function(x,u,Q,R) + ...
                    weights_critic'*vi_pf_critic(vi_controlled_system(x,u,A,B));
                u0 = weights_actor'*vi_pf_actor(x);
                ut = fminunc(objective,u0);
            end
            actor_target(:,j) = ut;
        end
        weights_actor = xu'\actor_target';
    end
    
    % update critic network
    if i == 1
        critic_target = vi_cost_function(x_train,0,Q,R)';
    else
        x_next = vi_controlled_system(x_train,weights_actor'*xu,A,B);
        critic_target = vi_cost_function(x_train,weights_actor'*xu,Q,R)'+ ...
                        (weights_critic'*vi_pf_critic(x_next))';
    end
    weights_critic = [ xV xV_minus]'\[ critic_target; critic_target];
    
    weights_actor_set = [weights_actor_set weights_actor];
    weights_critic_set = [weights_critic_set weights_critic];
    performance_index(i+1) = weights_critic'*vi_pf_critic(x0);
    figure(1),plot(i,performance_index(i+1),'*'),hold on;
    waitbar(i/epoch,h,['Training controller...',num2str(i/epoch*100),'%']);
end
close(h);

% check results
P_ini = [ weights_critic_ini(1) weights_critic_ini(2)/2; weights_critic_ini(2)/2 weights_critic_ini(3) ]

K_final = -weights_actor'
P_final = [ weights_critic(1) weights_critic(2)/2; weights_critic(2)/2 weights_critic(3) ]

[Kopt, Popt] = dlqr(A,B,Q,R)


figure(1),
xlabel('Iterations');
ylabel('Iterative $V(x_0)$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;
hold off;

figure(2),
plot((1:length(weights_critic_set))-1,weights_critic_set,'linewidth',1)
xlabel('Iterations');
ylabel('Critic NN weights'); 
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(3),
plot((1:length(weights_actor_set))-1,weights_actor_set,'linewidth',1)
xlabel('Iterations');
ylabel('Action NN weights'); 
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;


