function pi_algorithm
% This demo checks the feasibility of the policy iteration adaptive dynamic 
% programming algorithm for continuous-time systems

% By J. Lu
% Date: Nov. 27, 2020 

%-------------------------------- start -----------------------------------
clear; close all; clc;

% system information
A = [-0.0665   11.5       0         0;
         0     -2.5      2.5        0;
       -9.5      0    -13.736  -13.736;
        0.6      0       0         0];
B = [0; 0; 13.736; 0];

n = 4;
m = 1;

Q = eye(n); R = eye(m);

[~,P_opt] = lqr(A,B,Q,R);
[~,P_init] = lqr(A,B,Q,0.001*R); % obtain initial weights

w_opt = [ P_opt(1,1); 2*P_opt(1,2); 2*P_opt(1,3); 2*P_opt(1,4);...
          P_opt(2,2); 2*P_opt(2,3); 2*P_opt(2,4);...
          P_opt(3,3); 2*P_opt(3,4);...
          P_opt(4,4) ];

w_init = [ P_init(1,1); 2*P_init(1,2); 2*P_init(1,3); 2*P_init(1,4);...
           P_init(2,2); 2*P_init(2,3); 2*P_init(2,4);...
           P_init(3,3); 2*P_init(3,4);...
           P_init(4,4) ];

w = w_init;

ww = [ w ];

% generate training data
x0 = [ 1; -1; 0; 0 ];
x_train = [ x0 ];

for i = 1:100 
    x_train = [ x_train, zeros(n,1) ];
    x_train = [ x_train,10*(rand(n,1)-0.5) ];
    x_train = [ x_train,5*(rand(n,1)-0.5) ];
    x_train = [ x_train,2*(rand(n,1)-0.5) ];
    x_train = [ x_train,1*(rand(n,1)-0.5) ];
    x_train = [ x_train,0.5*(rand(n,1)-0.5) ];
    x_train = [ x_train,0.1*(rand(n,1)-0.5) ];
    x_train = [ x_train,0.05*(rand(n,1)-0.5) ];
    x_train = [ x_train,0.01*(rand(n,1)-0.5) ];
end

L = length(x_train);

epoch = 50;

h = waitbar(0,'please wait');
for k = 1:epoch 
    UU = zeros(L,1);
    XX = zeros(L,length(w));
    for i = 1:L
        x = x_train(:,i);

        dphi = [
                 [  2*x(1),          0,          0,          0  ];
                 [    x(2),       x(1),          0,          0  ];
                 [    x(3),          0,       x(1),          0  ];
                 [    x(4),          0,          0,       x(1)  ];
                 [       0,     2*x(2),          0,          0  ];
                 [       0,       x(3),       x(2),          0  ];
                 [       0,       x(4),          0,       x(2)  ];
                 [       0,          0,      2*x(3),         0  ];
                 [       0,          0,        x(4),      x(3)  ];
                 [       0,          0,          0,     2*x(4)  ];
            ];

        u = -0.5*inv(R)*B'*dphi'*w;

        UU(i) = x'*Q*x + u'*R*u;
        XX(i,:) = (dphi*(A*x + B*u))';
    end

    w = -XX\UU;

    ww = [ ww w ];
    
    waitbar(k/epoch,h,['Running...',num2str(k/epoch*100),'%']);
end
close(h);

% comparison
w_init_display = w_init'
w_final_display = w'
w_opt_display = w_opt'

figure(1), 
plot(((1:size(ww,2))-1),ww,'linewidth',1)
xlabel('Epoch');
ylabel('$w$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

end






