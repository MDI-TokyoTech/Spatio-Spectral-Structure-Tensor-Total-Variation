%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Article: Yao Wang, Jiangjun Peng, Qian Zhao, Yee Leung, Xi-Le Zhao, and Deyu Meng, 
%   ``Hyperspectral Image Restoration Via Total Variation Regularized Low-Rank Tensor Decomposition,''
%   IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2018.
%
% The code is Downloaded by 
% https://github.com/zhaoxile/Hyperspectral-Image-Restoration-via-Total-Variation-Regularized-Low-rank-Tensor-Decomposition
% and located at compared_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [HSI_restored, removed_noise] = func_LRTDTV(HSI_noisy, params)
%% Organizing for LRTDTV code
Noi = HSI_noisy;
tau = params.tau;
lambda = params.lambda;
rank = params.rank;

addpath(genpath('.\compared_methods\Hyperspectral-Image-Restoration-via-Total-Variation-Regularized-Low-rank-Tensor-Decomposition-master'))
% Remove quality_assess due to duplicate function names of ours

%% Running main function
[clean_image,S,~,~] = LRTDTV(Noi, tau,lambda,rank);


%% Organizing results for output
HSI_restored            = clean_image;
removed_noise.all_noise = S;
