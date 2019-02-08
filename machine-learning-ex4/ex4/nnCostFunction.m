function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1) X];
Y = zeros(m, num_labels);

for i=1:m
    for j=1:num_labels
        if y(i)==j
            Y(i,j)=1;
        else
            Y(i,j)=0;
        end
    end
end

for i=1:m
    a1 = X(i,:); % 1x401
    % fprintf('a1: %f ', a1);
    % fprintf('a1_1: %f ', size(a1,1));
    % fprintf('a1_2: %f ', size(a1,2));
    a2 = sigmoid(a1*Theta1');
    % fprintf('a2_1: %f ', size(a2,1));
    % fprintf('a2_2: %f ', size(a2,2));
    
    a2_ = [1 a2]; % 1x26
    a3 = sigmoid(a2_*Theta2'); % 1x10
    
    J = J + Y(i,:)*log(a3)'+(1-Y(i,:))*log(1-a3)';
    
    delta3 = a3-Y(i,:); % 1x10
    % fprintf('delta3: %f ', delta3);
    
    g = [1 sigmoidGradient(a1*Theta1')]; % 1X26
    delta2 = delta3*Theta2.*g; % 1X26
    % fprintf('delta2_1: %f ', size(delta2(2:end),1));
    % fprintf('delta2_2: %f ', size(delta2(2:end),2));
    
    Theta1_grad = Theta1_grad + delta2(2:end)'*a1;
    Theta2_grad = Theta2_grad + delta3'*a2_;
end

% A2 = [ones(m, 1) sigmoid(X*Theta1')];
% A3 = sigmoid(A2*Theta2'); % 5000X10

Theta1_ = Theta1;
Theta1_(:,1) = 0;
Theta2_ = Theta2;
Theta2_(:,1) = 0;

Reg = 0;
for i=1:size(Theta1_,1)
    for j=1:size(Theta1_,2)
        Reg = Reg + Theta1_(i,j)^2;
    end
end

for i=1:size(Theta2_,1)
    for j=1:size(Theta2_,2)
        Reg = Reg + Theta2_(i,j)^2;
    end
end

J = -1/m*(J-lambda/2*Reg);

Theta1_grad = 1/m*(Theta1_grad+lambda*Theta1_);
Theta2_grad = 1/m*(Theta2_grad+lambda*Theta2_);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
