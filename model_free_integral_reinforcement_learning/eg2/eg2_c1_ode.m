function xdot = eg2_c1_ode(~,x)

global M; global g; global L;
global k1; global k2; global k3;
global Ga;
global Q; global R; global Ra;
global Ta;
global dw;
global wa;

u = x(3);

dphi = eg2_c1_nn_pf_dphi(x(1:3));
    
v = -0.5*inv(Ra)*Ga'*dphi'*wa;

f = [x(2);-M*g*L/k1*sin(x(1))-k2/k1*x(2)+k3/k1*(tanh(u)+u)];

xdot = [ f; (1/Ta)*(v - x(3));...
         [x(1);x(2)]'*Q*[x(1);x(2)] + 1*log(1+[x(3)]'*R*[x(3)])+v'*Ra*v;...
         [x(1);x(2)]'*Q*[x(1);x(2)] + 1*log(1+[x(3)]'*R*[x(3)]); dw ];



end 