%% Spatio-Spectral Structure Tensor Total Variation for Hyperspectral Image Denoising and Destriping
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Shingo Takemoto (takemoto.s.e908@m.isct.ac.jp)
% Last version: June 15, 2025
% Article: S. Takemoto, K. Naganuma, S. Ono, 
%   ``Spatio-Spectral Structure Tensor Total Variation for Hyperspectral Image Denoising and Destriping''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear 
clc
close all
addpath(genpath('sub_functions'))
fprintf('******* initium *******\n');
rng('default')

%% Generating observation
%%%%%%%%%%%%%%%%%%%%% User settings of experiment %%%%%%%%%%%%%%%%%%%%%%%%%%%%
deg.Gaussian_sigma      = 0.1; % Standard derivation of Gaussian noise
deg.sparse_rate         = 0.05; % Rate of sparse noise
deg.stripe_rate         = 0.05; % Rate of stripe noise
deg.stripe_intensity    = 0.5; % Range of intensity for stripe noise

image = 'JasperRidge';
% image = 'PaviaUniversity';
% image = 'Beltsville';

show_band = 53;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

edge_width = 3;


dir_result_folder = append(...
    "./result/", ...
    "denoising_", image, "/", ...
    "g", num2str(deg.Gaussian_sigma), "_ps", num2str(deg.sparse_rate), ...
        "_pt", num2str(deg.stripe_rate), "/" ...   
);


%% Setting each methods info
% SSTV
methods_info(1) = struct( ...
    "name", "SSTV", ...
    "enable", true ...
);

% HSSTV_L1
methods_info(end+1) = struct( ...
    "name", "HSSTV_L1", ...
    "enable", true ...
);

% HSSTV_L12
methods_info(end+1) = struct( ...
    "name", "HSSTV_L12", ...
    "enable", true ...
);

% l0l1HTV
methods_info(end+1) = struct( ...
    "name", "l0l1HTV", ...
    "enable", true ...
);

% STV
methods_info(end+1) = struct( ...
    "name", "STV", ...
    "enable", true ...
);


% SSST
methods_info(end+1) = struct( ...
    "name", "SSST", ...
    "enable", true ...
);


% LRTDTV
methods_info(end+1) = struct( ...
    "name", "LRTDTV", ...
    "enable", true ...
);

% FGSLR
methods_info(end+1) = struct( ...
    "name", "FGSLR", ...
    "enable", true ...
);

% TPTV
methods_info(end+1) = struct( ...
    "name", "TPTV", ...
    "enable", true ...
);

% QRNN3D
methods_info(end+1) = struct( ...
    "name", "QRNN3D", ...
    "enable", false ...
);


% FastHyMix
methods_info(end+1) = struct( ...
    "name", "FastHyMix", ...
    "enable", true ...
);

% S3TTV (ours)
methods_info(1) = struct( ...
    "name", "S3TTV", ...
    "enable", true ...
);

methods_info = methods_info([methods_info.enable]); % removing false methods
num_methods = numel(methods_info);

i_method = 0;


vals_mpsnr = zeros(num_methods, 1);
vals_mssim = zeros(num_methods, 1);


%% Running methods
for idx_method = 1:num_methods
name_method = methods_info(idx_method).name;


dir_method_folder = fullfile(dir_result_folder, name_method);

names_params_tmp = dir(fullfile(dir_method_folder, '*.mat'));


names_params = {names_params_tmp.name};


best_name_params = 

fprintf("\n~~~ SETTINGS ~~~\n");
fprintf("Method: %s\n", name_method);
fprintf("Parameter settings: %s\n", name_params_savetext)
fprintf("Methods: (%d/%d), Params:(%d/%d)\n", ...
    idx_method, num_methods, idx_params_comb, num_params_comb);

[HSI_restored, ~] = func_method(HSI_noisy, params);


% Plotting results
val_mpsnr  = MPSNR(HSI_restored(:, :, edge_width+1:end-edge_width), HSI_clean(:, :, edge_width+1:end-edge_width));
val_mssim  = MSSIM(HSI_restored(:, :, edge_width+1:end-edge_width), HSI_clean(:, :, edge_width+1:end-edge_width));

fprintf("~~~ RESULTS ~~~\n");
fprintf("MPSNR: %#.4g\n", val_mpsnr);
fprintf("MSSIM: %#.4g\n", val_mssim);


save_folder_name = append(...
    "./result/", ...
    "denoising_", image, "/", ...
    "g", num2str(deg.Gaussian_sigma), "_ps", num2str(deg.sparse_rate), ...
        "_pt", num2str(deg.stripe_rate), "/", ...
    name_method, "/", ...
    name_params_savetext, "/" ...   
);

mkdir(save_folder_name);

save(append(save_folder_name, "restored_result.mat"), ...
    "HSI_restored", "val_mpsnr", "val_mssim", ...
    "-v7.3", "-nocompression" ...
);

close all

end
end

fprintf('******* finis *******\n');
