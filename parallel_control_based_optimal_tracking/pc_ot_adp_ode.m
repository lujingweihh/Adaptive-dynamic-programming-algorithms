function xdot = pc_ot_adp_ode(~,x)

global Ga;
global Q; global R; global Qa; global Ra;
global dw;
global w;
global a1; global a2;
global ra;
global ea;
global v;

w = x(6:end);

u = x(3);

fo_x = [-x(1)^3 + x(2); x(1)^2 - x(1) - x(2) + 0.15*u^3 + sin(0.1*u) ]; % nonaffine system

fo_r = [-ra(1)^3 + ra(2); ra(1)^2 - ra(1) - ra(2) + 0.15*ra(3)^3 + sin(0.1*ra(3)) ]; % reference signal

% construct the augmented system and reference signal first and derive the augmented tracking error system later 
% the paper obtains the tracking error system first and derives the augmented tracking error system later 
% both methods are acceptable
fa_e = [fo_x - fo_r; 0]; 

ea = x(1:3) - ra;

dphi = pc_ot_adp_nn_pf_dphi(ea(1:3));

v = -0.5*inv(Ra)*Ga'*dphi'*w;

dVe = ea'*(fa_e+Ga*v); % V = 0.5*(e1^2+e2^2+e3^2);

pVe = [ea(1);ea(2);ea(3)];

if dVe > 0
    k = 0.5;
end
if dVe <= 0
    k = 0;
end

sigma = dphi*(fa_e+Ga*v);
dw = -a1*sigma/((sigma'*sigma + 1)^2)*(sigma'*w + ea'*Qa*ea + v'*Ra*v)+k*a2*dphi*Ga*inv(Ra)*Ga'*pVe;

xdot = [ fo_x; v;...
         [ea(1);ea(2)]'*Q*[ea(1);ea(2)]+ea(3)'*R*ea(3);...
         [ea(1);ea(2);ea(3)]'*Qa*[ea(1);ea(2);ea(3)]+v'*Ra*v;...
         dw];

end 