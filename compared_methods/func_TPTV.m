%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Article: Yang Chen, Wenfei Cao, Li Pang, Jiangjun Peng, and Xiangyong Cao,
%   ``Hyperspectral Image Denoising Via Texture-Preserved Total Variation Regularizer,''
%   IEEE Transactions on Geoscience and Remote Sensing, 2023.
%
% The code is Downloaded by https://github.com/chuchulyf/ETPTV
% and located in compared_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [HSI_restored, removed_noise] = func_TPTV(HSI_noisy, params)
%% Organizing for TPTV code
Noi_H = HSI_noisy;
param = params;

Ori_H = HSI_noisy; % I don't use Ori_H and Xo

addpath(genpath('.\compared_methods\ETPTV-main'))


%% Running main function
[output_image,~,~,E,~] = WETV(Noi_H, Ori_H, param);


%% Organizing results for output
HSI_restored            = output_image;
removed_noise.all_noise = E;