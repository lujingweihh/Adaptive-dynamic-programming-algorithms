function dphi = eg1_c2_nn_pf_dphi(x)
dphi = [
         [  2*x(1),       0,         0   ];
         [    x(2),    x(1),         0   ];
         [    x(3),       0,      x(1)   ];
         [       0,  2*x(2),         0   ];
         [       0,    x(3),      x(2)   ];
         [       0,       0,    2*x(3)   ];
        ];
end