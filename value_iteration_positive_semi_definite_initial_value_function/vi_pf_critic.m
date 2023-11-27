function y = vi_pf_critic(x)
y = [   x(1,:).^2;   x(1,:).*x(2,:);   x(2,:).^2;   ];
end