%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Article: Yong Chen, Ting-Zhu Huang, Wei He, Xi-Le Zhao, Hongyan Zhang, Jinshan Zeng 
%   ``Hyperspectral Image Denoising Using Factor Group Sparsity-Regularized Nonconvex Low-Rank Approximation,''
%   IEEE Transactions on Geoscience and Remote Sensing, 2022.
%
% The code is Downloaded by https://chenyong1993.github.io/yongchen.github.io/
% and located at compared_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [HSI_restored, removed_noise] = func_FGSLR(HSI_noisy, params)
%% Organizing for FGSLR code
opt.alpha = 1;
opt.beta = params.beta;          % TV regularization:set as 0.1 or 0.5     
opt.lambda = 0.01;       % sparse noise
opt.mu = params.mu;              % penalty parameter:set as 5 or 10
opt.rho = 0.1;           % proximal parameter
opt.delta = params.delta;           % noise parameter:set as 0.5 or 5
opt.r = 20;
opt.regul_B = params.regul_B; % 'L2' or 'L21'

Img = HSI_noisy; % Dammy
Noisy_Img = HSI_noisy;

[M,N,p] = size(Noisy_Img);

addpath(genpath('.\compared_methods\FGSLR_code'))


%% Running main function
[X, ~, ~] = FGSLR_TV_PAM(Noisy_Img, Img, opt);
FGSLR_L2 = reshape(X',M,N,p);


%% Organizing results for output
HSI_restored = FGSLR_L2;

removed_noise.all_noise = HSI_noisy - HSI_restored;
