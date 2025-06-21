% Author: Shunsuke Ono (ono@isl.titech.ac.jp)
function[Du] = ProjL10ball(Du, epsilon)

sizeDu = size(Du);

Duin = Du;
% masking differences between opposite boundaries
Du(end,:,:,1) = 0;
Du(:,end,:,2) = 0;

sumDu = sum(sum(Du.^2, 4),3);
[v, I] = sort(sumDu(:), 'descend');
threInd = zeros(sizeDu(1:2));
threInd(I(1:epsilon)) = 1; % set ones for values to be held
threInd = repmat(threInd, [1 1 sizeDu(3:4)]);
Du = Du.*threInd;

Du(end,:,:,1) = Duin(end,:,:,1);
Du(:,end,:,2) = Duin(:,end,:,2);