clear
close all;

addpath('../../func_metrics/')

fprintf('******* initium *******\n');
rng('default')

%% Selecting conditions
noise_conditions = { ...
    {0.1,   0,      0,      0},     ... % g0.1 ps0 pt0
    {0,     0,      0.05,   0.5},   ... % g0 ps0 pt0.05
    {0.05,  0.05,   0,      0},     ... % g0.05 ps0.05 pt0
    {0.1,   0.05,   0,      0},     ... % g0.1 ps0.05 pt0
    {0.05,  0,      0.05,   0.5},   ... % g0.05 ps0 pt0.05
    {0.1,   0,      0.05,   0.5},   ... % g0.1 ps0 pt0.05
    {0.05,  0.05,   0.05,   0.5},   ... % g0.05 ps0.05 pt0.05
    {0.1,   0.05,   0.05,   0.5},   ... % g0.1 ps0.05 pt0.05
};

idc_noise_conditions = 1:size(noise_conditions, 2);
% idc_noise_conditions = 8;
% idc_noise_conditions = 1;
% idc_noise_conditions = 6;


images = {...
    'JasperRidge', ...
    'PaviaU120', ...
    'Beltsville', ...
};

idc_images = 1:numel(images);
% idc_images = 2;

idx_exp = 0;
total_exp = length(idc_noise_conditions) * length(idc_images);


for idx_image = idc_images
for idx_noise_condition = idc_noise_conditions
%% Generating observation
deg.gaussian_sigma      = noise_conditions{idx_noise_condition}{1};
deg.sparse_rate         = noise_conditions{idx_noise_condition}{2};
deg.stripe_rate         = noise_conditions{idx_noise_condition}{3};
deg.stripe_intensity    = noise_conditions{idx_noise_condition}{4};
image = images{idx_image};

% if idx_image == 1 && idx_noise_condition == 7 || ...
%         idx_image == 1 && idx_noise_condition == 8
%     deg.stripe_rate = 0.09;
% end

[HSI_clean, hsi] = Load_HSI_for_Deep(image);
[HSI_noisy, deg] = Generate_obsv_for_denoising(HSI_clean, deg);

% HSI_clean = single(HSI_clean);
% HSI_noisy = single(HSI_noisy);

idx_exp = idx_exp + 1;


%% Selecting methods
name_method = 'FastHyMix';

func_methods = {...
    @(params) func_FastHyMix(HSI_noisy, params), ...
};


%% Setting parameters
% FastHyMix
FastHyMix_k_subspace = {4, 6, 8, 10, 12};


%% Cordinating parameters
params_tmp = {FastHyMix_k_subspace};
name_params = {'k_subspace'};

[params_comb, num_params_comb] = ParamsList2Comb(params_tmp);


for idx_params_comb = 1:num_params_comb

params = struct();
for idx_params = 1:numel(name_params)
    % Assigning parameters to the structure
    params.(name_params{idx_params}) = params_comb{idx_params_comb}{idx_params};

end

name_params_savetext = append('sub', num2str(params.k_subspace));


%% Running methods

fprintf('\n~~~ SETTINGS ~~~\n');
fprintf('Method: %s\n', name_method);
fprintf('Image: %s Size: (%d, %d, %d)\n', image, hsi.n1, hsi.n2, hsi.n3);
fprintf('Gaussian sigma: %g\n', deg.gaussian_sigma);
fprintf('Sparse rate: %g\n', deg.sparse_rate);
fprintf('Stripe rate: %g\n', deg.stripe_rate);
fprintf('Stripe intensity: %g\n', deg.stripe_intensity);
fprintf('Parameter settings: %s\n', name_params_savetext)
fprintf('Cases: (%d/%d), Params:(%d/%d)\n', ...
    idx_exp, total_exp, idx_params_comb, num_params_comb);

try
    [HSI_restored, removed_noise, other_result]...
        = func_methods{1}(params);
catch
    fprintf('Error\n');
end


% Plotting results
mpsnr  = calc_MPSNR(HSI_restored, HSI_clean);
mssim  = calc_MSSIM(HSI_restored, HSI_clean);
ergas  = calc_ERGAS(HSI_restored, HSI_clean, 1); % GSD ratio = 1 for recovery problem;
sam    = calc_SAM(HSI_restored, HSI_clean);

fprintf('~~~ RESULTS ~~~\n');
fprintf('MPSNR: %#.4g\n', mpsnr);
fprintf('MSSIM: %#.4g\n', mssim);
fprintf('ERGAS: %#.4g\n', ergas);
fprintf('SAM  : %#.4g\n', sam);

[psnr_per_band, ssim_per_band] = calc_PSNR_SSIM_per_band(HSI_restored, HSI_clean);


% Saving each result
save_folder_name = append(...
    '../../result/' , ...
    'denoising_', image, '/', ...
    'g', num2str(deg.gaussian_sigma), '_ps', num2str(deg.sparse_rate), ...
        '_pt', num2str(deg.stripe_rate), '/', ...
    name_method, '/', ...
    name_params_savetext, '/' ...   
);

mkdir(save_folder_name);

save(append(save_folder_name, 'image_result.mat'), ...
    'HSI_clean', 'HSI_noisy', 'hsi', 'deg', 'image', ...
    'HSI_restored', 'removed_noise', ...
    '-v7.3', '-nocompression' ...
);

save(append(save_folder_name, 'metric_vals.mat'), ...
    'mpsnr', 'mssim', 'ergas', 'sam', ...
    'psnr_per_band', 'ssim_per_band', ...
    '-v7.3', '-nocompression' ...
);

save(append(save_folder_name, 'other_result.mat'), ...
    'params', 'other_result', ...
    '-v7.3', '-nocompression' ...
);

close all

end

end
end

fprintf('******* finis *******\n');