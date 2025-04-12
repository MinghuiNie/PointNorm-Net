function [vlue,vctr] = computePCA( X )

% X = X/(norm( max(X) ) - norm( min(X) ));
C = X'*X;
[vctr, vlue] = eig(C);

end

