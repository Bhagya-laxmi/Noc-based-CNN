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

%Part 1
a1= [ones(m,1) X]; %5000*401
size(a1);
a2 = sigmoid(a1 * Theta1'); %(5000*401) *(401*25)
a2= [ones(m,1) a2]; % 5000*25
size(a2);
a3= sigmoid(a2 * Theta2'); %(5000*25) * (26*10)
size(a3);

Y_modified = zeros(m, num_labels);%5000*10
for i=1: m
  Y_modified(i,y(i)) = 1;
end
 
J_wo = (1/m)* (sum(sum(-Y_modified .* log(a3) -(1 -Y_modified).*log(1- a3))));

regTheta1 =  Theta1(:,2:end);
regTheta2 =  Theta2(:,2:end);

J = J_wo + (lambda /(2*m))*(sum(sum(regTheta1.^2)) +sum(sum(regTheta2.^2)));


delta_3 = zeros(size(Theta2)); %10*26
delta_2 = zeros(size(Theta1)); %401*25

k = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);
%Part 2
for i= 1:m %1 to 5000
  
  %Step 1:
  a_1 = X(i,:); %1 example at a time 1*400
  size(a_1);
  a_1 = [1 a_1]; %1*401
  a_2 = sigmoid(a_1*Theta1'); %1*25
  a_2 =[1 a_2]; %1*26
  size(a_2);
  a_3 = sigmoid(a_2*Theta2'); %1*10
  size(a_3);
  %Step 2
  delta_3 = a_3-Y_modified(i,:); %1*10
  size(delta_3);
  
  %Step 3 
  z2= [1 a_1*Theta1']; %1*26
  check1 = sigmoidGradient(z2);
  delta_2 = (delta_3*Theta2).*sigmoidGradient(z2); %1*26
  
  %Step 4
  delta_2 = delta_2(2:end); %1*25  
  size(delta_2);
  Theta1_grad = Theta1_grad + delta_2' *(a_1); 
  
  Theta2_grad = Theta2_grad + delta_3'*(a_2);
  
 %%%%%%%%%%%%%%%%%%%%%%%%%%%
% del1 = zeros(size(Theta1));
% del2 = zeros(size(Theta2));
%   
% a1t = a1(i,:);
% a2t = a2(i,:);
% a3t = a3(i,:);
%   
% yt = k(i,:);
% d3 = a3t - yt;
% check = sigmoidGradient([1;Theta1 * a1t']);
% d2 = Theta2'*d3' .* sigmoidGradient([1;Theta1 * a1t']);
% del1 = del1 + d2(2:end)*a1t;
% del2 = del2 + d3' * a2t;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%
  

end

% Step 5
Theta2_grad = (1/m) * Theta2_grad; % (10*26)
Theta1_grad = (1/m) * Theta1_grad; % (25*401)


%With regularization
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)* Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)* Theta2(:,2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];



end
