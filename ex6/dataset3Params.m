function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
values = [0.01 0.03 0.1 0.3 1 3 10 30];

[CCand sigmaCand] = meshgrid(values,values);
candidate_values = [CCand(:) sigmaCand(:)];

C = candidate_values(1,1);
sigma = candidate_values(1,2);
tempModel = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
tempPred = svmPredict(tempModel,Xval);
acc = mean(double(tempPred == yval));
fprintf('initial accuracy: %d.\n', acc);

for i = 2:size([CCand(:) sigmaCand(:)],1)
  tempC = candidate_values(i,1);
  tempSigma = candidate_values(i,2);
  
  tempModel = svmTrain(X, y, tempC, @(x1, x2) gaussianKernel(x1, x2, tempSigma));
  tempPred = svmPredict(tempModel,Xval);
  tempAcc = mean(double(tempPred == yval));
  
  if tempAcc > acc
    C = tempC;
    sigma = tempSigma;
    acc = tempAcc
  endif
  
endfor





% =========================================================================

end
