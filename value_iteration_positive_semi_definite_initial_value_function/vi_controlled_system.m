function x_next = vi_controlled_system(x,u,A,B)
%--------------------------- controlled system ----------------------------
x_next = A*x + B*u;
end