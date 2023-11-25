function xdot = eg1_c1_parallel_ode(~,x)

global A; global B; 
global Q; global R;
global Ta;
global Kaopt;
global ua;

ua = x(3);

v = -Kaopt*[x(1);x(2);x(3)];

xdot = [A*[x(1);x(2)]+B*ua; (1/Ta)*(v - x(3)); [x(1);x(2)]'*Q*[x(1);x(2)]+ua'*R*ua];

end