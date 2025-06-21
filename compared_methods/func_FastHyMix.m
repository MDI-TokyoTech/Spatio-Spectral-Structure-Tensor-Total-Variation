%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Article: Yang Chen, Wenfei Cao, Li Pang, Jiangjun Peng, and Xiangyong Cao,
%   ``Hyperspectral Image Denoising Via Texture-Preserved Total Variation Regularizer,''
%   IEEE Transactions on Geoscience and Remote Sensing, 2023.
%
% The code is Downloaded by https://github.com/chuchulyf/ETPTV
% and located in compared_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [HSI_restored, removed_noise] = func_FastHyMix(HSI_noisy, params)
addpath('./HSI-MixedNoiseRemoval-FastHyMix-main/scripts');
k_subspace = params.k_subspace;

img_noisy = HSI_noisy;

dir_prev = pwd;
cd('./compared_methods/HSI-MixedNoiseRemoval-FastHyMix-main/')

[img_FastHyMix, ~, ~] = FastHyMix(img_noisy,  k_subspace);

cd(dir_prev)

%% Organizing results for output
HSI_restored = img_FastHyMix;

removed_noise.all_noise = HSI_noisy - HSI_restored;

end

