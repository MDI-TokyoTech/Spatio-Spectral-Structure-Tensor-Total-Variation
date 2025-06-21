%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Shingo Takemoto (takemoto.s.e908@m.isct.ac.jp)
% Last version: June 15, 2025
% Article: Minghua, Wang, Qiang Wang, Jocelyn Chanussot, and Danfeng Hong, 
%   ``l0-l1 Hybrid Total Variation Regularization and Its Applications on 
%       Hyperspectral Image Mixed Noise Removal and Compressed Sensing,''
%   IEEE Transactions on Geoscience and Remote Sensing, 2021.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% f(U,S,T) = |D(Ds(U))|_1 + l_{1,0}(D(U)) + L1ball(T) + 
%               L2ball(U+S+T) + box constraint(U) + Dv(T)=0
%
% f1(U,S,T) = 0
% f2(U,S,T) = box constraint(U) + L1ball(S) + L1ball(T)
% f3(U,S,T) = |D(Ds(U))|_1 + l_{1,0}(D(U)) + L2ball(U+S+T) + Dv(T)=0
%
% A = (DDs O O; D O O; I I I; O O Dv)
%
% Algorithm is based on P-PDS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [HSI_restored, removed_noise, iteration, converge_rate_U] ...
     = l0l1HTV_GPU(HSI_noisy, params)
fprintf('** Running l0l1HTV_GPU **\n');
HSI_noisy   = gpuArray(single(HSI_noisy));
[n1, n2, n3] = size(HSI_noisy);

epsilon             = gpuArray(single(params.epsilon));
L10ball_th          = params.L10ball_th;
alpha               = gpuArray(single(params.alpha));
beta                = gpuArray(single(params.beta));
stepsize_reduction  = gpuArray(single(params.stepsize_reduction));
maxiter             = gpuArray(single(params.maxiter));
stopcri             = gpuArray(single(params.stopcri));

%% Setting params
disprate    = gpuArray(single(1000));


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
% Y1: term of SSTV
% Y2: term of l_{1,0} norm
% Y3: term of l2ball
% Y4: term of stripe noise
%
% Y1 = D(Ds(U))
% Y2 = l_{1,0}(D(U))
% Y3 = U + S + T
% Y4 = Dv(T)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Y1 = zeros([n1, n2, n3, 2], 'single', 'gpuArray');
Y2 = zeros([n1, n2, n3, 2], 'single', 'gpuArray');
Y3 = zeros([n1, n2, n3], 'single', 'gpuArray');
Y4 = zeros([n1, n2, n3], 'single', 'gpuArray');


%% Setting operators
D       = @(z) cat(4, z([2:end, 1],:,:) - z, z(:,[2:end, 1],:) - z);
Dt      = @(z) z([end,1:end-1],:,:,1) - z(:,:,:,1) + z(:,[end,1:end-1],:,2) - z(:,:,:,2);
Dv      = @(z) z([2:end, 1],:,:) - z;
Dvt     = @(z) z([end,1:end-1],:,:) - z(:,:,:);
Dh      = @(z) z(:,[2:end, 1],:) - z;
Ds      = @(z) z(:, :, [2:end, 1], :) - z;
Dst     = @(z) z(:,:,[end,1:end-1],:) - z(:,:,:,:);

% Determining parameta "theta" in L0gradient
HSI_mean_3 = mean(HSI_noisy, 3);
diff_mean = (abs(Dv(HSI_mean_3)) + abs(Dh(HSI_mean_3)))/2;
% diff_mean = (abs(Dv_circulant(HSI_mean_3)) + abs(Dh_circulant(HSI_mean_3)))/2;

diff_mean(diff_mean < L10ball_th) = 0; 
% gamma_prime = nnz(diff_mean)/(128*128);
% theta = floor(128*128*gamma_prime);
eps_L10ball = gpuArray(single(nnz(diff_mean)));


%% Setting stepsize parameters for P-PDS
gamma1_U    = gpuArray(single(1./(2*2 + 2*2 + 2 + 2 + 1)));
gamma1_S    = gpuArray(single(1));
gamma1_T    = gpuArray(single(1/(2 + 1)));
gamma2_Y1   = gpuArray(single(1/(2*2)));
gamma2_Y2   = gpuArray(single(1/2));
gamma2_Y3   = gpuArray(single(1/3));
gamma2_Y4   = gpuArray(single(1/2));


%% main loop (P-PDS)
converge_rate_U = zeros([1, maxiter], 'single');
fprintf('~~~ P-PDS STARTS ~~~\n');

for i = 1:maxiter
    tic;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating U
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    U_tmp   = U - gamma1_U.*(Dst(Dt(Y1)) + Dt(Y2) + Y3);
    U_next  = ProjBox(U_tmp, 0, 1);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating S
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    S_tmp   = S - gamma1_S.*Y3;
    S_next  = ProjFastL1Ball(S_tmp, alpha);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating T
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    T_tmp   = T - gamma1_T.*(Y3 + Dvt(Y4));
    T_next  = ProjFastL1Ball(T_tmp, beta);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating Y1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Y1_tmp  = Y1 + gamma2_Y1.*(D(Ds(2*U_next - U)));
    Y1_next = Y1_tmp - gamma2_Y1.*Prox_l1norm(Y1_tmp./gamma2_Y1, 1/gamma2_Y1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating Y2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Y2_tmp  = Y2 + gamma2_Y2.*(D(2*U_next - U));
    Y2_next = Y2_tmp - gamma2_Y2.*ProjL10ball(Y2_tmp./gamma2_Y2, eps_L10ball);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating Y3
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Y3_tmp  = Y3 + gamma2_Y3.*(2*(U_next + S_next + T_next) - (U + S + T));    
    Y3_next = Y3_tmp - gamma2_Y3.*ProjL2ball(Y3_tmp./gamma2_Y3, HSI_noisy, epsilon);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating Y4
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Y4_next = Y4 + gamma2_Y4.*Dv(2*T_next - T);

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
    Y4  = Y4_next;

    gamma1_U    = gamma1_U  * stepsize_reduction;
    gamma1_S    = gamma1_S  * stepsize_reduction;
    gamma1_T    = gamma1_T  * stepsize_reduction;
    gamma2_Y1   = gamma2_Y1 * stepsize_reduction;
    gamma2_Y2   = gamma2_Y2 * stepsize_reduction;
    gamma2_Y3   = gamma2_Y3 * stepsize_reduction;
    gamma2_Y4   = gamma2_Y4 * stepsize_reduction;

 
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
