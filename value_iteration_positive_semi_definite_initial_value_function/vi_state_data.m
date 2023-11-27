%-------------------------- generate training data ------------------------
clear; close all; clc;

A = [0,0.1; 0.3, -1]; B = [0;0.5];
Q = 1*eye(2); R = 1;
control_dim = 1;
state_dim = 2;
x0 = [1;-1];

Jcompression = 1;
ucompression = 1;

x_train = zeros(state_dim,1);
u_train = zeros(control_dim,1);

for i = 1:30
    x_train = [x_train, zeros(state_dim,1)];
    x_train = [x_train,4*(rand(state_dim,1)-0.5)];   % [-2 2]
    x_train = [x_train,2*(rand(state_dim,1)-0.5)];   % [-1 1]
    x_train = [x_train,1*(rand(state_dim,1)-0.5)];   % [-0.5 0.5]
    x_train = [x_train,0.2*(rand(state_dim,1)-0.5)]; % [-0.1 0.1]
    u_train = [u_train,zeros(control_dim,1)];
    u_train = [u_train,4*(rand(control_dim,1)-0.5)];   % [-2 2]
    u_train = [u_train,2*(rand(control_dim,1)-0.5)];   % [-1 1]
    u_train = [u_train,1*(rand(control_dim,1)-0.5)];   % [-0.5 0.5]
    u_train = [u_train,0.2*(rand(control_dim,1)-0.5)]; % [-0.1 0.1]
end

r = randperm(size(x_train,2));   % randomization according column
x_train = x_train(:,r);         % reorder
[~,n] = find(x_train == zeros(state_dim,1));

zeros_index = [];
for i = 1:size(n,1)/state_dim
    zeros_index = [zeros_index, n(state_dim*i)];
end

ru = randperm(size(u_train,2));   % randomization according column
u_train = u_train(:,ru);         % reorder

xu_train = [x_train; u_train];
x_next_train = A*x_train + B*u_train;

save state_data x_train u_train xu_train x_next_train zeros_index x0 control_dim state_dim A B Q R




