function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

P = 0;
for i = 2:size(theta)
    P = theta(i)*theta(i)+P
end

for i = 1:m
    J = y(i)*log(1/(1+exp(X(i,:)*theta*(-1))))+(1-y(i))* ...
        log(1-1/(1+exp(X(i,:)*theta*(-1))))+J;
end

J = -1/m*(J-lambda/2*P);

for j = 1:size(theta)
    for i = 1:m
        grad(j)=(1/(1+exp(X(i, :)*theta*(-1)))-y(i))*X(i,j) + grad(j);     
    end
    if j > 1
        grad(j) = grad(j)+lambda*theta(j);
    end
end

grad=1/m*grad;


% =============================================================

end
