function xdot = irl_ode(~,x)
% parameters
global K; global u; global A; global B; global Q; global R;

x = [x(1) x(2) x(3) x(4)]';

% calculate the control signal
u = -K*x;

xdot = [A*x + B*u; x'*Q*x + u'*R*u];