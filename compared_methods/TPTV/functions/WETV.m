%% solve the problem
%                       min_X ||U_x||_1+||U_y||_1 +||U_z||_1 +||E||_1
%                         s.t. D_x(X) = U_x*V_x, V_x'*V_x=I
%                              D_y(X) = U_y*V_y, V_y'*V_y=I
%                              D_z(X) = U_z*V_z, V_z'*V_z=I
%                              Y      = X +E ,rank(U_x,y,z)=r
%                          ===============================
%                          ===============================
%         D is difference operator, T is difference tensor,T is known
%  ------------------------------------------------------------------------


function [ output_image,U_x,V_x,E,Xo] = WETV(Noi_H,Ori_H, param)

[M,N,p] = size(Noi_H);
sizeD   = size(Noi_H);

if (~isfield(param,'maxIter'))
    maxIter = 100;
else
    maxIter = param.maxIter;
end

if (~isfield(param,'tol'))
    tol = 1e-6;
else
    tol = param.tol;
end

if (~isfield(param,'rho'))
    rho     = 1.2;
else
    rho     = param.rho;
end

if (~isfield(param,'lambda'))
    lambda = 1;
else
    lambda = param.lambda;
end

if (~isfield(param,'Alpha'))
    Alpha = 1;
else
    Alpha = param.Alpha;
end

if (~isfield(param,'lr_init'))
    lr_init = 'SVD';
else
    lr_init = param.lr_init;
end

if (~isfield(param,'initial_rank'))
    r0 = 1;
else
    r0 = param.initial_rank;
end

if (~isfield(param,'Rank'))
    r = ceil(0.3*p*ones(1,3));
else
    r = param.Rank;
end


max_mu   = 1e6;
mu      = 1e-2;
D        = zeros(M*N,p) ;

%%
Y       = reshape(Noi_H,M*N,p);
for i=1:p
    bandp = Noi_H(:,:,i);
    D(:,i)= bandp(:);
end
normD   = norm(D,'fro');

%% FFT setting
h               = sizeD(1);
w               = sizeD(2);
d               = sizeD(3);
%%
Eny_x   = ( abs(psf2otf([+1; -1], [h,w,d])) ).^2  ;
Eny_y   = ( abs(psf2otf([+1, -1], [h,w,d])) ).^2  ;
Eny_z   = ( abs(psf2otf([+1, -1], [w,d,h])) ).^2  ;
Eny_z   =  permute(Eny_z, [3, 1 2]);
determ  =  Eny_x + Eny_y + Eny_z;

%% Initializing optimization variables
% X              = lr_init;
X              = reshape(Noi_H,M*N,p);
E              = Y - X;
M1 =zeros(size(D));  % multiplier for D-X-E
M2 =zeros(size(D));  % multiplier for Dx_X-U_x*V_x
M3 =zeros(size(D));  % multiplier for Dy_X-U_y*V_y
M4 =zeros(size(D));  % multiplier for Dz_X-U_z*V_z

% main loop
iter = 0;
while iter<maxIter
    iter          = iter + 1;
    
    %% weight of TV image
    if mod(iter,5)==0
        r0 = r0+1;
    end
    
    [u0, s0, v0] = svd(reshape(X,M*N,p), 'econ');
    X0        = u0(:,1:r0)*s0(1:r0,1:r0)*v0(:,1:r0)';
    
    W_s            = zeros(M*N,3);
    tmp_dx         = reshape(diff_x(X0,sizeD),[M*N,p]);
    [u,s,~]        = svd(tmp_dx,'econ');
    U_x            = u(:,1:r(1));
    
    tmp_dy         = reshape(diff_y(X0,sizeD),[M*N,p]);
    [u,s,~]        = svd(tmp_dy,'econ');
    U_y            = u(:,1:r(2));
    
    tmp_dz         = reshape(diff_z(X0,sizeD),[M*N,p]);
    [u,s,~]        = svd(tmp_dz,'econ');
    U_z            = u(:,1:r(3));
    
    Y_dx = abs(mean(tmp_dx,2));
    Y_dy = abs(mean(tmp_dy,2));
    Y_dz = abs(mean(tmp_dz,2));
    
    W_s(:,1) = Y_dx(:)/max(Y_dx(:));
    W_s(:,2) = Y_dy(:)/max(Y_dy(:));
    W_s(:,3) = Y_dz(:)/max(Y_dz(:));
    clear W_s1 W_s2 W_s3 Y_dx Y_dy Y_dz
    W_s      = 1./(W_s + 5e-2);
    W_s      = W_s / max(W_s(:));
%              show3Dimg(reshape(W_s,M,N,3))
    %    show3Dimg(reshape(abs([U_x,U_y,U_z]),M,N,sum(r)))
    %% -Update V_x and V_y and V_z
    tmp_x         = tmp_dx+M2/mu;
    [u,s,v]       = svd(tmp_x,'econ');
    U_x           = u(:,1:r(1))*s(1:r(1),1:r(1));
    V_x           = ( v(:,1:r(1))' )';
    
    tmp_y         = tmp_dy+M3/mu;
    [u,s,v]       = svd(tmp_y,'econ');
    U_y           = u(:,1:r(2))*s(1:r(2),1:r(2));
    V_y           = ( v(:,1:r(2))' )';
    
    tmp_z         = tmp_dz+M4/mu;
    [u,s,v]       = svd(tmp_z,'econ');
    U_z           = u(:,1:r(3))*s(1:r(3),1:r(3));
    V_z           = ( v(:,1:r(3))' )';
    %      show3Dimg(reshape(abs([U_x,U_y,U_z]),M,N,sum(r)))
    %% -Update U_x and U_y and U_z
    
    weight        = repmat(W_s(:,1),[1,r(1)]);
    U_x           = softthre(U_x, lambda/mu*weight);
    weight        = repmat(W_s(:,2),[1,r(2)]);
    U_y           = softthre(U_y, lambda/mu*weight);
    weight        = repmat(W_s(:,3),[1,r(3)]);
    U_z           = softthre(U_z, lambda/mu*weight);
    %       show3Dimg(reshape(abs([U_x,U_y,U_z]),M,N,sum(r)))
    
    %% -Updata X
    % solve TV by FFT algorithm: relatively low accurary, less cost time
    diffT_p  = diff_xT(mu*U_x*V_x'-M2,sizeD)+diff_yT(mu*U_y*V_y'-M3,sizeD);
    diffT_p  = diffT_p + diff_zT(mu*U_z*V_z'-M4,sizeD);
    numer1   = reshape( diffT_p + mu*(D(:)-E(:)) + M1(:), sizeD);
    x        = real( ifftn( fftn(numer1) ./ (mu*determ + mu) ) );
    X        = reshape(x,[M*N,p]);
    %        show3Dimg(reshape(X,M,N,p))
            Xo(iter)=  norm(Ori_H(:)-X(:),'fro')/norm(Ori_H(:),'fro');
    %% -Update E
    E             = softthre(D-X+M1/mu, 1/mu);
    %      show3Dimg(reshape(E,M,N,p))
    %     E               = (M1+mu*(D-X))/(2*lambda+mu);% Gaussian noise
    %% stop criterion
    leq1 = D -X -E;
    leq2 = reshape(diff_x(X,sizeD),[M*N,p])- U_x*V_x';
    leq3 = reshape(diff_y(X,sizeD),[M*N,p])- U_y*V_y';
    leq4 = reshape(diff_z(X,sizeD),[M*N,p])- U_z*V_z';
    stopC1 = norm(leq1,'fro')/normD;
    stopC2 = max(abs(leq2(:)));
    stopC4 = norm(leq4,'fro')/normD;
    
    %     disp(['iter ' num2str(iter) ',mu=' num2str(mu1,'%2.1e')  ...
    %             ',Y-X-E=' num2str(stopC1,'%2.3e') ',||DX-UV||=' num2str(stopC2,'%2.3e')...
    %             ',|DZ-UV|' num2str(stopC4,'%2.3e')]);
    if stopC1<tol && stopC2<tol
        break;
    else
        M1 = M1 + mu*leq1;
        M2 = M2 + mu*leq2;
        M3 = M3 + mu*leq3;
        M4 = M4 + mu*leq4;
        mu = min(max_mu,mu*rho);
    end
    %     load('Simu_indian.mat');
    %     [mp(iter),sm(iter),er(iter)]=msqia(simu_indian,reshape(X,[M,N,p]));
end
output_image = reshape(X,[M,N,p]);


end

