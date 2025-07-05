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

show_band = 53; % Select band for show
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch image
    case 'JasperRidge'
        load('./dataset/JasperRidge.mat');

    case 'PaviaUniversity'
        load('./dataset/PaviaUniversity.mat');

    case 'Beltsville'
        load('./dataset/Beltsville.mat');
end

[HSI_noisy, deg] = Generate_obsv(HSI_clean, deg);

image_clean = HSI_clean(:,:,show_band);
image_noisy = HSI_noisy(:,:,show_band);

[n1, n2] = size(image_clean);


edge_width = 3;

fprintf("\n~~~ CONDITION SETTINGS ~~~\n");
fprintf("Image: %s \n", image);
fprintf("Gaussian sigma: %g\n", deg.Gaussian_sigma);
fprintf("Sparse rate: %g\n", deg.sparse_rate);
fprintf("Stripe rate: %g\n", deg.stripe_rate);
fprintf("Stripe intensity: %g\n", deg.stripe_intensity);


dir_result_folder = append(...
    "./result/", ...
    image, "/", ...
    "g", num2str(deg.Gaussian_sigma), "_ps", num2str(deg.sparse_rate), ...
        "_pt", num2str(deg.stripe_rate), "/" ...   
);


%% Setting each methods info
% SSTV
methods_info(1) = struct( ...
    "name", "SSTV", ...
    "enable", false ...
);

% HSSTV_L1
methods_info(end+1) = struct( ...
    "name", "HSSTV1", ...
    "enable", false ...
);

% HSSTV_L12
methods_info(end+1) = struct( ...
    "name", "HSSTV2", ...
    "enable", false ...
);

% l0l1HTV
methods_info(end+1) = struct( ...
    "name", "l0l1HTV", ...
    "enable", false ...
);

% STV
methods_info(end+1) = struct( ...
    "name", "STV", ...
    "enable", false ...
);


% SSST
methods_info(end+1) = struct( ...
    "name", "SSST", ...
    "enable", false ...
);


% LRTDTV
methods_info(end+1) = struct( ...
    "name", "LRTDTV", ...
    "enable", false ...
);

% FGSLR
methods_info(end+1) = struct( ...
    "name", "FGSLR", ...
    "enable", false ...
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
    "enable", false ...
);

% S3TTV (ours)
methods_info(1) = struct( ...
    "name", "S3TTV", ...
    "enable", false ...
);

methods_info = methods_info([methods_info.enable]); % removing false methods
num_methods = numel(methods_info);

i_method = 0;


%% Choosing best paramters for each method
% Initialiging
vals_mpsnr = zeros(num_methods, 1);
vals_mssim = zeros(num_methods, 1);

cat_images = zeros([n1, n2, num_methods+2]);
cat_images(:,:,1) = image_clean;
cat_images(:,:,2) = image_noisy;

names_params_best = cell(num_methods, 1);


for idx_method = 1:num_methods
    name_method = methods_info(idx_method).name;
    fprintf("\n~~ Choosing the parameters for %s ~~\n", name_method);
    
    
    dir_method_folder = fullfile(dir_result_folder, name_method);
    
    names_params_tmp = dir(fullfile(dir_method_folder, '*.mat'));
    names_params = {names_params_tmp.name};
    
    
    val_max_mpsnr = 0;
    name_params_best = strings(1);
    
    % Searching best parameters for mpsnr
    for i = 1:numel(names_params)
        name_params = names_params{i};
    
        load(fullfile(dir_method_folder, name_params), "val_mpsnr");
    
    
        if val_max_mpsnr < val_mpsnr
            val_max_mpsnr = val_mpsnr;
            name_params_best = name_params;
        end
    end

    fprintf("Best paramter of %s: %s", name_method, name_params_best);
    
    
    % Extracting best result
    load(fullfile(dir_method_folder, name_params_best));

    vals_mpsnr(idx_method)          = val_mpsnr;
    vals_mssim(idx_method)          = val_mssim;
    cat_images(:,:,idx_method+2)    = HSI_restored(:,:,show_band);
    names_params_best{idx_method}   = name_params_best;
end


%% Plotting evaluation results
% Preparing
name_length_max = max(strlength([methods_info.name]));
name_params_length_max = max(strlength(names_params_best));

fprintf("\n\n~~~ SETTINGS ~~~\n");
fprintf("Image: %s Size: (%d, %d, %d)\n", image, hsi.n1, hsi.n2, hsi.n3);
fprintf("Gaussian sigma: %g\n", deg.Gaussian_sigma);
fprintf("Sparse rate: %g\n", deg.sparse_rate);
fprintf("Stripe rate: %g\n", deg.stripe_rate);
fprintf("Stripe intensity: %g\n", deg.stripe_intensity);


fprintf("~~~ RESULTS ~~~\n");
fprintf("%s  \t MPSNR\t MSSIM\n", blanks(name_length_max + name_params_length_max + 2));


for idx_method = 1:num_methods
    name_method = methods_info(idx_method).name;
    name_params_best = names_params_best{idx_method};

    fprintf("%s(%s): \t %#.4g\t %#.4g\n", ...
        append(name_method, blanks(name_length_max - strlength(name_method))), ...
        append(name_params_best, blanks(name_params_length_max - strlength(name_params_best))), ...
        val_mpsnr, val_mssim);
end


%% Showing result images
figure;
subplot(1, num_methods+2, 1)
imshow(image_clean)
title("GT")

subplot(1, num_methods+2, 2)
imshow(image_noisy)
title("Noisy")


for idx_method = 1:num_methods
    name_method = methods_info(idx_method).name;

    subplot(1, num_methods+2, idx_method+2)
    imshow(cat_images(:,:,idx_method+2))
    title(name_method)
end


fprintf('******* finis *******\n');
