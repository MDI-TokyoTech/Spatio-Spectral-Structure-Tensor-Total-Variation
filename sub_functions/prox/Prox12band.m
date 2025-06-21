% this function is prox of sum_{k}^{n3} ||X(:, :, k)||_{1,2} + bandway
% suppose X to be 4D tensor
% X has variation of row in 1st 3D tensor and colunun in 2nd 3D tensor

function result = Prox12band(X, gamma)
    
    T = max(1 - gamma./sqrt(sum(sum(X.*X, 4), 3)), 0);
    result = T.*X;

end