%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Shingo Takemoto (takemoto.s.e908@m.isct.ac.jp)
% Last version: June 15, 2025
% Article: Ryuji Kurihara, Shunsuke Ono, Keiichiro Shirai, and Masahiro Okuda, 
%   ``Hyperspectral Image Restoration Based on Spatio-Spectral Structure Tensor Regularization,''
%   in Proc. Eur. Signal Process. Conf. (EUSIPCO), 2017.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% f(U,S,T) = Σ_{k=1}^{K}|P_k(D(U))|_* + 
%           L1ball(S) + L1ball(T) + 
%           L2ball(U+S+T) + box constraint(U) + Dv(T)=0
%
% f1(U,S,T) = 0
% f2(U,S,T) = box constraint(U) + L1ball(S) + L1ball(T)
% f3(U,S,T) = |P(D(U))|_*,N + L2ball(U+S+T) + Dv(T)=0
%
% A = (PD O O; O O Dv; I I I)
%
% Algorithm is based on P-PDS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [HSI_restored, removed_noise, iteration, converge_rate_U] ...
     = SSST_GPU(HSI_noisy, params)
fprintf('** Running SSST_GPU **\n');
HSI_noisy   = gpuArray(single(HSI_noisy));
[n1, n2, n3] = size(HSI_noisy);

alpha       = gpuArray(single(params.alpha));
beta        = gpuArray(single(params.beta));
epsilon     = gpuArray(single(params.epsilon));
blocksize   = gpuArray(single(params.blocksize));
maxiter     = gpuArray(single(params.maxiter));
stopcri     = gpuArray(single(params.stopcri));

%% Setting params
disprate    = gpuArray(single(100));

%% Initializing primal and dual variables

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% primal variables
% U: clean HSI
% S: sparse noise(salt-and-pepper noise)
% T: stripe noise
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

U = zeros([n1, n2, n3], 'single', 'gpuArray');
S = zeros([n1, n2, n3], 'single', 'gpuArray');
T = zeros([n1, n2, n3], 'single', 'gpuArray');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dual variables
% Y1: term of S3TTV
% Y2: term of l2ball
% Y3: term of stripe noise
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Y1 = zeros([n1, n2, n3, 3, blocksize(1), blocksize(2)], 'single', 'gpuArray');
Y2 = zeros([n1, n2, n3], 'single', 'gpuArray');
Y3 = zeros([n1, n2, n3], 'single', 'gpuArray');


%% Setting operators
% Difference operators
D       = @(z) cat(4, z([2:end, 1],:,:) - z, z(:,[2:end, 1],:) - z, z(:,:,[2:end, 1]) - z);
Dt      = @(z) z([end,1:end-1],:,:,1) - z(:,:,:,1) ...
                + z(:,[end,1:end-1],:,2) - z(:,:,:,2) ...
                + z(:,:,[end,1:end-1],3) - z(:,:,:,3);
Dv      = @(z) z([2:end, 1],:,:) - z;
Dvt     = @(z) z([end,1:end-1],:,:) - z(:,:,:);

% Expansion operators
P = @(z) func_PeriodicExpansion(z, blocksize);
Pt = @(z) func_PeriodicExpansionTrans(z);


%% Setting stepsize parameters for P-PDS
gamma1_U    = gpuArray(single(1./(prod(blocksize) * 2*3 + 1)));
gamma1_S    = gpuArray(single(1));
gamma1_T    = gpuArray(single(1/(2 + 1)));
gamma2_Y1   = gpuArray(single(1/2));
gamma2_Y2   = gpuArray(single(1/3));
gamma2_Y3   = gpuArray(single(1/2));


%% main loop (P-PDS)
converge_rate_U = zeros([1, maxiter], 'single');
fprintf('~~~ P-PDS STARTS ~~~\n');

for i = 1:maxiter   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating U
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    U_tmp   = U - gamma1_U.*(Dt(Pt(Y1)) + Y2);
    U_next  = ProjBox(U_tmp, 0, 1);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating S
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    S_tmp   = S - gamma1_S.*Y2;
    S_next  = ProjFastL1Ball(S_tmp, alpha);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating T
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    T_tmp   = T - gamma1_T.*(Y2 + Dvt(Y3));
    T_next  = ProjFastL1Ball(T_tmp, beta);

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating Y1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Y1_tmp  = Y1 + gamma2_Y1.*(P(D(2*U_next - U)));
    Y1_next = Y1_tmp - gamma2_Y1.*Prox_S3TTV(Y1_tmp./gamma2_Y1, 1./gamma2_Y1, blocksize);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating Y2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Y2_tmp  = Y2 + gamma2_Y2.*(2*(U_next + S_next + T_next) - (U + S + T));    
    Y2_next = Y2_tmp - gamma2_Y2.*ProjL2ball(Y2_tmp./gamma2_Y2, HSI_noisy, epsilon);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating Y3
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Y3_next = Y3 + gamma2_Y3.*Dv(2*T_next - T);

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculating error
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    converge_rate_U(i) = norm(U_next(:) - U(:),2)/norm(U(:),2);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating all variables
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    U   = U_next;
    S   = S_next;
    T   = T_next;
    
    Y1  = Y1_next;
    Y2  = Y2_next;
    Y3  = Y3_next;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Convergence checking
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if i>=2 && converge_rate_U(i) < stopcri
        fprintf('Iter: %d, Error: %0.6f.\n', i, converge_rate_U(i));
        break
    end
    if (mod(i, disprate) == 0) % Displaying intermediate results
        fprintf('Iter: %d, Error: %0.6f.\n', i, converge_rate_U(i));
    end
end

fprintf('~~~ P-PDS ENDS ~~~\n');

%% Organizing results for output
HSI_noisy                       = gather(HSI_noisy);
HSI_restored                    = gather(U);
iteration                       = gather(i);
removed_noise.sparse_noise      = gather(S);
removed_noise.stripe_noise      = gather(T);
removed_noise.gaussian_noise    = HSI_noisy - HSI_restored - ...
                                    removed_noise.sparse_noise - removed_noise.stripe_noise;
converge_rate_U                 = gather(converge_rate_U(1:iteration));
