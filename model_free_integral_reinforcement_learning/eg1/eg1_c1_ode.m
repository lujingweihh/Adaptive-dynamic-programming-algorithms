function xdot = eg1_c1_ode(~,x)

global A; global B;
global Q; global R;
global Kopt;
global u;

u = -Kopt*[x(1);x(2)];

xdot = [A*[x(1);x(2)]+B*u; [x(1);x(2)]'*Q*[x(1);x(2)]+u'*R*u];

end 