function xdot = eg1_c2_ode(~,x)

global A; global B;
global Ba;
global Qa; global Ra;
global Ta;
global dw;
global wa;

u = x(3);

dphi = eg1_c2_nn_pf_dphi(x(1:3));
    
v = -0.5*inv(Ra)*Ba'*dphi'*wa;

xdot = [A*[x(1);x(2)]+B*u; (1/Ta)*(v - x(3)); [x(1);x(2);x(3)]'*Qa*[x(1);x(2);x(3)]+v'*Ra*v;dw];

end 