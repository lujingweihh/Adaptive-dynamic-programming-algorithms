%------------------------- generate training data & system information ----------------------------
clear; close all; clc;

% system matrices
A = [  0,      0.1;...
       0.3,    -1   ];
B = [  0;...
       0.5  ];

state_dim = size(A,1);
control_dim = size(B,2);

% cost function parameters
Q = 1*eye(state_dim);
R = 1*eye(control_dim);

% training data
x_train = zeros(state_dim,1);
x0 = [1;-1];

for i = 1:50
    x_train = [x_train, zeros(state_dim,1)];  
    x_train = [x_train,2*(rand(state_dim,1)-0.5)]; 
    x_train = [x_train,1*(rand(state_dim,1)-0.5)];
    x_train = [x_train,0.5*(rand(state_dim,1)-0.5)];
end

r = randperm(size(x_train,2));   % randomization according to column
x_train = x_train(:, r);         % reorder

save training_data/state_data x_train state_dim control_dim A B Q R x0;




