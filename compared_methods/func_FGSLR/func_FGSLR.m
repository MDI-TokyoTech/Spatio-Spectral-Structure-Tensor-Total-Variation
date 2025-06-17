function [HSI_restored, removed_noise, other_result] = func_FGSLR(HSI_clean, HSI_noisy, params)
addpath FGSLR_PAM
addpath quality_assess

opt.alpha = 1;
opt.beta = params.beta;          % TV regularization:set as 0.1 or 0.5     
opt.lambda = 0.01;       % sparse noise
opt.mu = params.mu;              % penalty parameter:set as 5 or 10
opt.rho = 0.1;           % proximal parameter
opt.delta = params.delta;           % noise parameter:set as 0.5 or 5
opt.r = 20;
opt.regul_B = params.regul_B; % 'L2' or 'L21'



Img = HSI_clean;
Noisy_Img = HSI_noisy;

[M,N,p] = size(Noisy_Img);

tic
[X, A, B] = FGSLR_TV_PAM(Noisy_Img, Img, opt);
FGSLR_L2 = reshape(X',M,N,p);

runtime = toc;
fprintf("Runtime: %g sec\n", runtime);


%% Organizing results for output
HSI_restored = FGSLR_L2;

removed_noise.all_noise = HSI_noisy - HSI_restored;

other_result.A = A;
other_result.B = B;
end

