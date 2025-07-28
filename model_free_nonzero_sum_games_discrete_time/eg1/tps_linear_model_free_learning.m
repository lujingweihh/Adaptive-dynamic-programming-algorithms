function tps_linear_model_free_learning
% ''Learning-Based Parallel Control for Unknown 
%   Nonaffine Nonzero-Sum Gamesl''
%
% Code for Example 1
%
% By J. Lu
% Date: Oct. 22, 2024

%-------------------------------- start -----------------------------------
clear; close all; clc;

% performance index & system dynamics
state_dim = 2;
control_dim = 1;

Q1 = 1*eye(state_dim);
Q2 = 2*eye(state_dim);
R11 = 2*eye(control_dim);
R12 = 2*eye(control_dim);
R21 = 3*eye(control_dim);
R22 = 3*eye(control_dim);
Ra11 = 1*eye(control_dim);
Ra12 = 0*eye(control_dim);
Ra21 = 0*eye(control_dim);
Ra22 = 1*eye(control_dim);

A = [ 0.9950 0.0998; -0.0998 0.9950 ];
B1 = [ 0.2097; 0.0898 ];
B2 = [ 0.2147; 0.2895 ]; 

Ba1 = [ zeros(state_dim,control_dim);...
        eye(control_dim);...
        zeros(control_dim,control_dim) ];
Ba2 = [ zeros(state_dim,control_dim);...
        zeros(control_dim,control_dim);...
        eye(control_dim) ];

% initial states & controls
x0 = [1; -1];
u10 = 0;
u20 = 0;
s0 = [x0; u10; u20];

x = x0;
u1 = u10;
u2 = u20;

% learning parameters
rho_c1 = 1e-9;
rho_c2 = 1e-9;

rho_a1_initial = 0.01;
rho_a2_initial = 0.01;

L = 49;

critic_update_flag = 0;
actor_update_flag = 0;

ssc = [];
ssc_next = [];
rrac1 = [];
rrac2 = [];

wc1 = [ 1  0  0  0   ...
        1  0  0   ... 
        1  0   ... 
        1 ]'; 

wc2 = wc1;

wa1 = [ -1;  0.5; -1;  0 ];
wa2 = [ -1; -0.5;  0; -1 ];

xx = [ x ];
uu1 = [ u1 ];
uu2 = [ u2 ];
wwc1 = [ wc1 ];
wwc2 = [ wc2 ];
wwa1 = [ wa1 ];
wwa2 = [ wa2 ];

N = 20000;

decay_interval = 2000;

rho_a1 = rho_a1_initial;
rho_a2 = rho_a2_initial;
rho_a1_final = 1e-5; 
rho_a2_final = 1e-5; 
rho_a1_decay = (rho_a1_initial-rho_a1_final)/(0.8*N/decay_interval);
rho_a2_decay = (rho_a2_initial-rho_a2_final)/(0.8*N/decay_interval);

h = waitbar(0,'Please wait');
for k = 1:N
    if mod(k,decay_interval) == 0
        rho_a1 = rho_a1 - rho_a1_decay;
        rho_a2 = rho_a2 - rho_a2_decay;
    end

    if rho_a1 < rho_a1_final
        rho_a1 = rho_a1_final;
    end

    if rho_a2 < rho_a2_final
        rho_a2 = rho_a2_final;
    end

    if norm(x) <= 0.1 % reset controls to excite the 2-player system (TPS)
        u1 = 10*(rand(1)-0.5);
        u2 = 10*(rand(1)-0.5);
    end

    s = [x;u1;u2];

    v1 = wa1'*actor_afv(s);
    v2 = wa2'*actor_afv(s);

    r = utility_function(x,u1,u2,Q1,Q2,R11,R12,R21,R22);
    ra = utility_function_augmented(r,v1,v2,Ra11,Ra12,Ra21,Ra22);

    if size(ssc,2) < L
        ssc = [ssc s];
        rrac1 = [rrac1 ra(1)];
        rrac2 = [rrac2 ra(2)];
    else
        if length(ssc) >= L
            ssc_temp = [ ssc(:,2:end) s ];
            ssc = ssc_temp;

            rrac1_temp = [ rrac1(:,2:end) ra(1) ];
            rrac1 = rrac1_temp;

            rrac2_temp = [ rrac2(:,2:end) ra(2)];
            rrac2 = rrac2_temp;
        end
    end

    x = system_dynamics(x,u1,u2,A,B1,B2);

    u1 = u1 + v1;
    u2 = u2 + v2;

    s_next = [x;u1;u2];

    if size(ssc_next,2) < L
        ssc_next = [ssc_next s_next];
    else
        if size(ssc_next,2) >= L
            ssc_next_temp = [ ssc_next(:,2:end) s_next ];
            ssc_next = ssc_next_temp;
        end
    end

    if size(ssc_next,2) == L
        critic_update_flag = 1;
        actor_update_flag = 1;
    else
        critic_update_flag = 0;
    end

    if norm(ssc_next) > 0.1 && critic_update_flag == 1
        wc1 = critic_weights(ssc,ssc_next,rrac1,wc1,rho_c1);   
        wc2 = critic_weights(ssc,ssc_next,rrac2,wc2,rho_c2);        
    end

    if actor_update_flag == 1
        wa1 = actor_weights(s,s_next,wc1,wa1,rho_a1,Ra11,Ba1);
        wa2 = actor_weights(s,s_next,wc2,wa2,rho_a2,Ra22,Ba2);
    end

    xx = [ xx x ];
    uu1 = [ uu1 u1 ];
    uu2 = [ uu2 u2 ];
    wwc1 = [ wwc1 wc1 ];
    wwc2 = [ wwc2 wc2 ];
    wwa1 = [ wwa1 wa1 ];
    wwa2 = [ wwa2 wa2 ];

    waitbar(k/N,h,['Running...',num2str(k/N*100),'%']);
end
close(h);

wc1_display = wc1'
wc2_display = wc2'

wa1_display = wa1'
wa2_display = wa2'

figure(1), % states 
plot(((1:size(xx,2))-1),xx,'linewidth',1);
xlabel('$k$','Interpreter','latex');
ylabel('$x$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(2), % controls
plot(((1:size(uu1,2))-1),uu1,'linewidth',1),hold on;
plot(((1:size(uu2,2))-1),uu2,'linewidth',1),hold off;
xlabel('$k$','Interpreter','latex');
ylabel('$u$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(3), % wc1
plot(((1:size(wwc1,2))-1),wwc1,'linewidth',1);
xlabel('$k$','Interpreter','latex');
ylabel('$\mathcal{W}_{c,1}$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(4), % wc2
plot(((1:size(wwc2,2))-1),wwc2,'linewidth',1);
xlabel('$k$','Interpreter','latex');
ylabel('$\mathcal{W}_{c,2}$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(5), % wa1
plot(((1:size(wwa1,2))-1),wwa1,'linewidth',1);
xlabel('$k$','Interpreter','latex');
ylabel('$\mathcal{W}_{a,1}$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(6), % wa2
plot(((1:size(wwa2,2))-1),wwa2,'linewidth',1);
xlabel('$k$','Interpreter','latex');
ylabel('$\mathcal{W}_{a,2}$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

end
%--------------------------- system dynamics ------------------------------
function x_next = system_dynamics(x,u1,u2,A,B1,B2)

x_next = A*x + B1*u1 + B2*u2;

end
%--------------------------- utility function -----------------------------
function r = utility_function(x,u1,u2,Q1,Q2,R11,R12,R21,R22)

r = [ x'*Q1*x + u1'*R11*u1 + u2'*R12*u2;...
      x'*Q2*x + u1'*R21*u1 + u2'*R22*u2 ];

end
%----------------------- augmented utility function -----------------------
function ra = utility_function_augmented(r,v1,v2,Ra11,Ra12,Ra21,Ra22)

ra = r + [ v1'*Ra11*v1 + v2'*Ra12*v2;...
           v1'*Ra21*v1 + v2'*Ra22*v2 ];

end
%------------------ activation function vector of critic ------------------
function afv = critic_afv(x)
x1 = x(1);
x2 = x(2);
x3 = x(3);
x4 = x(4);

afv = [ x1^2; x1*x2; x1*x3; x1*x4;...
        x2^2; x2*x3; x2*x4;...
        x3^2; x3*x4;...
        x4^2 ];

end
%------------------- activation function vector of actor ------------------
function afv = actor_afv(x)
x1 = x(1);
x2 = x(2);
x3 = x(3);
x4 = x(4);

afv = [ x1; x2; x3; x4 ];

end
%-------------- derivative of critic activation function vector -----------
function dtheta = critic_afv_derivative(x)
x1 = x(1);
x2 = x(2);
x3 = x(3);
x4 = x(4);

dtheta = [
            [  2*x1,          0,          0,          0  ];
            [    x2,         x1,          0,          0  ];
            [    x3,          0,         x1,          0  ];
            [    x4,          0,          0,         x1  ];
            [     0,       2*x2,          0,          0  ];
            [     0,         x3,         x2,          0  ];
            [     0,         x4,          0,         x2  ];
            [     0,          0,       2*x3,          0  ];
            [     0,          0,         x4,         x3  ];
            [     0,          0,          0,       2*x4  ];
];

end
%----------------------------- update critic ------------------------------
function wc_next = critic_weights(ss,ss_next,rra,wc,rho_c)
varrho = [];
varphi = [];

for j = size(ss,2):-1:1
    varphi = [ varphi critic_afv(ss_next(:,j)) - critic_afv(ss(:,j)) ];
end

for j = size(rra,2):-1:1
    varrho = [ varrho rra(j) ];
end

zeta_c = varrho + wc'*varphi;

wc_next = varphi*pinv(varphi'*varphi)*(rho_c*zeta_c'- varrho');

end
%----------------------------- update actor -------------------------------
function wa_next = actor_weights(s,s_next,wc,wa,rho_a,Ra,Ga)

theta = actor_afv(s);

dtheta = critic_afv_derivative(s_next);

ve = wa'*theta + 0.5*pinv(Ra)*Ga'*dtheta'*wc;

wa_next = wa - rho_a*theta*ve'/(theta'*theta+1);

end


