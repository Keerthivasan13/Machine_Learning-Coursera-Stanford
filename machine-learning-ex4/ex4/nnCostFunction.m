function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m,1) X];
J = 0;
big_delta_1 = zeros(size(Theta1));
big_delta_2 = zeros(size(Theta2));
for i = 1:m
a1 = X(i,:)';
z2 = Theta1 * a1;
a2 = [1;sigmoid(z2)];
z3 = Theta2 * a2;
a3 = sigmoid(z3);

y_t = zeros(num_labels,1);
y_t(y(i),1) = 1;
J = J - ((y_t' * log(a3)) + ((1 - y_t)' * log(1 - a3)));

delta_3 = a3 - y_t;
delta_2 = (Theta2(:,2:end)' * delta_3) .* (sigmoidGradient(z2));

big_delta_1 = big_delta_1 + delta_2 * a1';
big_delta_2 = big_delta_2 + delta_3 * a2';
end
J = J / m;
Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta1_grad = (big_delta_1 + lambda .* Theta1) ./ m;
Theta2_grad = (big_delta_2 + lambda .* Theta2) ./ m;

t1 = (Theta1 .^ 2);
t2 = (Theta2 .^ 2);
regular = (lambda / (2 * m)) * (sum(t1(:)) + sum(t2(:)));

J = J + regular;
%%
% $x^2+e^{\pi i}$
% -------------------------------------------------------------

% ============
%%
% | | _MONOSPACED TEXT_ | | =============================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
