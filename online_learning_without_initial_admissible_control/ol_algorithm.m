function ol_algorithm
% This demo checks the feasibility of online leranbing without 
% initial admissible control

% By J. Lu
% Date: Mar. 16, 2021

%-------------------------------- start -----------------------------------
clear; close all; clc;

global u;
global Q; global R; 
global probing_noise;

Q = 1*eye(2); R = 1;

wc_ini = [0.1;0;0.1];
wc = wc_ini;

length_nn = length(wc_ini);

x_ini = [1;-1];
x0 = [x_ini; wc]';
xx = [x0'];
uu = [];
wwc = [wc];

% simulation settings
Fsamples = 5000;
T = 0.1; 
ss = T/10; 
tspan = 0:ss:T;

% generate the probing noise
A = 10;
noise = sqrt(A)*randn(1,Fsamples/T);

N = length(noise);
n = 1;

h = waitbar(0,'please wait');
for k = 1:Fsamples
    probing_noise = noise(n);

    [t,x]= ode45(@spiol_ode, tspan, x0);

    x0 = x(length(t),:);
    xx = [xx x0'];
    uu = [uu u];
    wwc = [wwc, x0(3:end)'];

    n = n + 1;

    if n > N
        n = 1;
    end

    waitbar(k/Fsamples,h,['Running...',num2str(k/Fsamples*100),'%']);
end
close(h)

% check results
wc_ini = wc_ini'
wc_final = x0(3:end)

wc_opt = [0.5 0 1]

% results
figure(1), % states 
plot(T*((1:size(xx,2))-1),xx(1:2,:),'linewidth',1)
xlabel('Time (s)');
ylabel('$x$','Interpreter','latex');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(2), % control
plot(T*((1:size(uu,2))-1),uu,'b','linewidth',1)
xlabel('Time (s)');
ylabel('Control');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

figure(3), % critic weights
plot(T*((1:size(wwc,2))-1),wwc,'linewidth',1)
xlabel('Time (s)');
ylabel('Critic weights');
set(gca,'FontName','Times New Roman','FontSize',14,'linewidth',1);
grid on;

end

%-------------------------------- ode -------------------------------------
function sdot = spiol_ode(~,s)
global u;
global Q; global R;
global probing_noise;

% learning parameters
a1 = 25;
a2 = 0.01;

x = s(1:2);
wc = s(3:end);

f = [ -x(1)+x(2);...
      -x(1)/2-x(2)*(1-(cos(2*x(1)+2)^2))/2 ];

g = [ 0;...
      cos(2*x(1)+2)];

pphi = pd_phi(x);

ux = -0.5*inv(R)*g'*pphi'*wc;

% candidate Lyapunov function V = 0.5*(x1^2+x2^2);
pV = [ x(1); x(2) ];

dV = x'*(f+g*u);

if dV > 0
    k = 0.5;
else
    k = 0;
end

sigma = pphi*(f+g*ux);

dwc = -a1*sigma/((sigma'*sigma+1)^2)*(sigma'*wc+x'*Q*x+ux'*R*ux)...
                 + k*a2*pphi*g*inv(R)*g'*pV;

u = ux + probing_noise;

sdot = [ f+g*u; dwc ];
end 
%------------- partial derivative of activation function vector -----------
function pphi = pd_phi(x)
pphi = [ 2*x(1)     0;...
           x(2)     x(1);...
           0      2*x(2)];
end 




