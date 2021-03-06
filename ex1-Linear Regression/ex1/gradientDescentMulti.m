function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    theta_k = zeros(size(theta));
    for k = 1:size(theta)
        theta_k(k) = theta(k);
    end
    for j = 1:length(theta)
        J = 0;
        for i = 1:m
            h_theta = 0;
            for k = 1:size(X,2)
                h_theta = h_theta + theta_k(k) * X(i,k);
            end
            J = J + (h_theta - y(i)) * X(i,j);
        end
        theta(j) = theta(j) - alpha / m * J;
    end
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
