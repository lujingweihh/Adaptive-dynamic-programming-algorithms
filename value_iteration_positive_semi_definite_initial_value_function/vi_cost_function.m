function y = vi_cost_function(x,u,Q,R)
%----------------------------- cost function ------------------------------
y = (diag(x'*Q*x) + diag(u'*R*u))';
end