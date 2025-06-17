clear
close all;

addpath('func_metrics')

fprintf('******* initium *******\n');
rng('default')

%% Selecting conditions
noise_conditions = { ...
    {0.1, 0, 0, 0}, ... % g0.1 ps0 pt0
    {0, 0, 0.05, 0.5}, ... % g0 ps0 pt0.05
    {0.05, 0.05, 0, 0}, ... % g0.05 ps0.05 pt0
    {0.1, 0.05, 0, 0}, ... % g0.1 ps0.05 pt0
    {0.05, 0, 0.05, 0.5}, ... % g0.05 ps0 pt0.05
    {0.1, 0, 0.05, 0.5}, ... % g0.1 ps0 pt0.05
    {0.05, 0.05, 0.05, 0.5}, ... % g0.05 ps0.05 pt0.05
    {0.1, 0.05, 0.05, 0.5}, ... % g0.1 ps0.05 pt0.05
};

idc_noise_conditions = 1:size(noise_conditions, 2);
% idc_noise_conditions = 1:2;
% idc_noise_conditions = 8;


images = {...
    'JasperRidge', ...
    'PaviaU120', ...
    'Beltsville', ...
};

idc_images = 1:numel(images);
% idc_images = 3;


for idx_noise_condition = idc_noise_conditions
for idx_image = idc_images
%% Generating observation
deg.gaussian_sigma      = noise_conditions{idx_noise_condition}{1};
deg.sparse_rate         = noise_conditions{idx_noise_condition}{2};
deg.stripe_rate         = noise_conditions{idx_noise_condition}{3};
deg.stripe_intensity    = noise_conditions{idx_noise_condition}{4};
image = images{idx_image};

[HSI_clean, hsi] = Load_HSI_for_ref_code(image);
[HSI_noisy, deg] = Generate_obsv_for_denoising(HSI_clean, deg);

% HSI_clean = single(HSI_clean);
% HSI_noisy = single(HSI_noisy);


%% Selecting methods
name_method = 'FGSLR';

func_methods = {...
    @(params) func_FGSLR(HSI_clean, HSI_noisy, params), ...
};


%% Setting parameters
% FGSLR
FGSLR_beta_set = [0.1];
FGSLR_mu_set = [5];
FGSLR_delta_set = [0.5];
FGSLR_regul_B_set = {'L2'};
idc_FGSLR_regul_B = 1:numel(FGSLR_regul_B_set);

for FGSLR_beta = FGSLR_beta_set
for FGSLR_mu = FGSLR_mu_set
for FGSLR_delta = FGSLR_delta_set
for idx_FGSLR_regul_B = idc_FGSLR_regul_B
FGSLR_regul_B = FGSLR_regul_B_set{idx_FGSLR_regul_B};

%% Running methods
params_tmp = {FGSLR_beta, FGSLR_mu, FGSLR_delta, FGSLR_regul_B};
name_params = {'beta', 'mu', 'delta', 'regul_B'};
name_params_savetext = append('b', num2str(FGSLR_beta), '_m', num2str(FGSLR_mu), ...
        '_d', num2str(FGSLR_delta), '_', FGSLR_regul_B);


fprintf('~~~ SETTINGS ~~~\n');
fprintf('Method: %s\n', name_method);
fprintf('Image: %s Size: (%d, %d, %d)\n', image, hsi.n1, hsi.n2, hsi.n3);
fprintf('Gaussian sigma: %g\n', deg.gaussian_sigma);
fprintf('Sparse rate: %g\n', deg.sparse_rate);
fprintf('Stripe rate: %g\n', deg.stripe_rate);
fprintf('Stripe intensity: %g\n', deg.stripe_intensity);

if name_method == 'FGSLR'
    fprintf('beta: %g\n', FGSLR_beta);
    fprintf('mu: %g\n', FGSLR_mu);
    fprintf('delta: %g\n', FGSLR_delta);
    fprintf('norm: %s\n', FGSLR_regul_B);
end


params = struct();
for idx_params = 1:numel(params_tmp)
    params.(name_params{idx_params}) = params_tmp{idx_params};
end

[HSI_restored, removed_noise, other_result]...
    = func_methods{1}(params);


% % Plotting results
% mpsnr  = calc_MPSNR(HSI_restored, HSI_clean);
% mssim  = calc_MSSIM(HSI_restored, HSI_clean);
% ergas  = calc_ERGAS(HSI_restored, HSI_clean, 1); % GSD ratio = 1 for recovery problem;
% sam    = calc_SAM(HSI_restored, HSI_clean);
% 
% fprintf('~~~ RESULTS ~~~\n');
% fprintf('MPSNR: %#.4g\n', mpsnr);
% fprintf('MSSIM: %#.4g\n', mssim);
% fprintf('ERGAS: %#.4g\n', ergas);
% fprintf('SAM  : %#.4g\n', sam);
% 
% [psnr_per_band, ssim_per_band] = calc_PSNR_SSIM_per_band(HSI_restored, HSI_clean);
% 
% 
% % Saving each result
% save_folder_name = append(...
%     '../../result/' , ...
%     'denoising_', image, '/', ...
%     'g', num2str(deg.gaussian_sigma), '_ps', num2str(deg.sparse_rate), ...
%         '_pt', num2str(deg.stripe_rate), '/', ...
%     name_method, '/', ...
%     name_params_savetext, '/' ...   
% );
% 
% mkdir(save_folder_name);
% 
% save(append(save_folder_name, 'image_result.mat'), ...
%     'HSI_clean', 'HSI_noisy', 'hsi', 'deg', 'image', ...
%     'HSI_restored', 'removed_noise', ...
%     '-v7.3', '-nocompression' ...
% );
% 
% save(append(save_folder_name, 'metric_vals.mat'), ...
%     'mpsnr', 'mssim', 'ergas', 'sam', ...
%     'psnr_per_band', 'ssim_per_band', ...
%     '-v7.3', '-nocompression' ...
% );
% 
% save(append(save_folder_name, 'other_result.mat'), ...
%     'params', 'other_result', ...
%     '-v7.3', '-nocompression' ...
% );

close all

end
end
end
end

end
end

fprintf('******* finis *******\n');