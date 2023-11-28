function puol_algorithm
% This demo checks the feasibility of policy update online learning

% By J. Lu
% Date: Apr. 16, 2021

%-------------------------------- start -----------------------------------
clear; close all; clc;

A = [0 -0.8;0.8 1.8]; B = [0; -1];
Q = eye(2); R = 1;

ac = 10e-6;
aa = 0.1;
wc = zeros(8,1);
wa = [0.5;1.4;0;0;0;0];
wa_ini = wa;
Nc = 4;

x = 5*[(rand(1,1)-0.5);(rand(1,1)-0.5)];
xx = [x];
xxc = [];
rrc = [];
uu = [];
wwc = [wc];
wwa = [wa];

N = 10000;

h = waitbar(0,'please wait');
for k = 1:N
    if length(xxc) < Nc
        u = actor(wa,actor_activation_function(x));
    else
        if length(xxc) == Nc
            wa = actor_weights(x,actor(wa,actor_activation_function(x)),wa,aa,wc,A,B,Q,R);
            u = actor(wa,actor_activation_function(x));
        end
    end

    r = x'*Q*x+u'*R*u;

    % update system states
    x = controlled_system(x,u,A,B);
    
    xx = [xx x];
    uu = [uu u];
    
    if length(xxc) < Nc
        xxc = [xxc x];
        rrc = [rrc r];
    else
        if length(xxc) == Nc
            xxc_temp = xxc(:,2:end);
            xxc_temp = [xxc_temp x];
            xxc = xxc_temp;
            
            rrc_temp = rrc(:,2:end);
            rrc_temp = [rrc_temp r];
            rrc = rrc_temp;
        end
    end
    
    if length(xxc) == Nc 
        wc = critic_weights(xxc,rrc,wc,ac);
    end
    
    if norm(x,2)<10e-5  % states converge to 0 and reset the initial states
        % usually, it is necessary to change controls to change system states
        % since the paper does not explain how to excite the system
        % a simple reset of initial states is used
        x = 5*[(rand(1,1)-0.5);(rand(1,1)-0.5)];
        xxc = [];
        rrc = [];
    end

    wwc = [wwc wc];
    wwa = [wwa wa];
    
    waitbar(k/N,h,['Running...',num2str(k/N*100),'%']);
end
close(h);

% check results
wa_ini = wa_ini'

wc_final = wc'
wa_final = wa'

[Kopt, Popt] = dlqr(A,B,Q,R);

wc_opt = [ Popt(1,1) 2*Popt(1,2) Popt(2,2) ]
wa_opt = [-Kopt(1,1) -Kopt(1,2)]


figure(1), % states 
plot(((1:size(xx,2))-1),xx,'linewidth',1)
xlabel('Time steps');
ylabel('States');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(2), % control
plot(((1:size(uu,2))-1),uu,'b','linewidth',1)
xlabel('Time steps');
ylabel('Control');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(3), % critic weights
plot(((1:size(wwc,2))-1),wwc,'linewidth',1)
xlabel('Time (s)');
ylabel('Critic weights');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(4), % actors weights
plot(((1:size(wwa,2))-1),wwa,'linewidth',1)
xlabel('Time (s)');
ylabel('Action weights');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

end

%--------------------------- outpout of system ----------------------------
function y = controlled_system(x,u,A,B)
y = A*x + B*u; 
end
%-------------------------------- actor -----------------------------------
function y = actor(wa,oa)
y = wa'*oa;
end
%----------------------- critic activation function -----------------------
function y = critic_activation_function(x)
y = [ x(1)^2;             x(1)*x(2);      x(2)^2;...
      x(1)^4;             x(1)^3*x(2);    x(1)^2*x(2)^2;...
      x(1)*x(2)^3;        x(2)^4 ];
end
%----------------------- action activation function -----------------------
function y = actor_activation_function(x)
y = [ x(1) x(2) (x(1)^2)*x(2) x(1)^3 x(1)*x(2)^2 x(2)^3]';
end
%------------------------- update critic weights --------------------------
function wc = critic_weights(xx,rr,w,ac)
X = [];
Y = [];

for k = length(xx):-1:2
    X = [ X critic_activation_function(xx(:,k)) -...
            critic_activation_function(xx(:,k-1)) ];
end

for j = length(rr):-1:2
    Y = [ Y rr(j)];
end

E = Y + w'*X;
wc = X*pinv(X'*X)*(ac*E'-Y');
end
%-------------------------- update actor weights --------------------------
function wa = actor_weights(x,u0,w,aa,wc,A,B,Q,R)
rho = actor_activation_function(x);

objective = @(u) x'*Q*x+u'*R*u + ...
                 wc'*critic_activation_function(controlled_system(x,u,A,B));
uopt = fminunc(objective,u0);

ue =  w'*rho - uopt;

wa = w - aa*rho*ue/(rho'*rho+1);
end














