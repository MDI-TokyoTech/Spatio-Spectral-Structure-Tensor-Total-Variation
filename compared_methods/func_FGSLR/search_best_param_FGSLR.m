clear;
close all;

addpath(genpath('sub_functions'))


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


images = {...
    'JasperRidge', ...
    'PaviaU120', ...
    'Beltsville', ...
};

% idc_images = 1:numel(images);
idc_images = 3;


for idx_noise_condition = idc_noise_conditions
for idx_image = idc_images
%% Selecting noise condition
deg.gaussian_sigma      = noise_conditions{idx_noise_condition}{1};
deg.sparse_rate         = noise_conditions{idx_noise_condition}{2};
deg.stripe_rate         = noise_conditions{idx_noise_condition}{3};
deg.stripe_intensity    = noise_conditions{idx_noise_condition}{4};
image = images{idx_image};


[HSI_clean, hsi] = Load_HSI_for_ref_code(image);
[HSI_noisy, deg] = Generate_obsv_for_denoising(HSI_clean, deg);

HSI_clean = single(HSI_clean);
HSI_noisy = single(HSI_noisy);


diff_magnification = 7;


fprintf('~~~ SETTINGS ~~~\n');
fprintf('Image: %s Size: (%d, %d, %d)\n', image, hsi.n1, hsi.n2, hsi.n3);
fprintf('Gaussian sigma: %g\n', deg.gaussian_sigma);
fprintf('Sparse rate: %g\n', deg.sparse_rate);
fprintf('Stripe rate: %g\n', deg.stripe_rate);
fprintf('Stripe intensity: %g\n', deg.stripe_intensity);


%% Setting parameters
% FGSLR
% FGSLR_beta_set = [0.1, 0.5];
FGSLR_beta_set = [0.5];
FGSLR_mu_set = [5, 10];
FGSLR_delta_set = [0.5, 5];
FGSLR_regul_B_set = {'L2', 'L21'};
idc_FGSLR_regul_B = 1:numel(FGSLR_regul_B_set);


%% Initialiging best param
best_beta = [];
best_mu = [];
best_delta = [];
best_regul_B = [];

best_mpsnr = 0;

i = 0;

fprintf('~~~ RESULTS ~~~\n');
fprintf('     \t MPSNR\t MSSIM\t SAM\n');


%% Comparing each param
for FGSLR_beta = FGSLR_beta_set
for FGSLR_mu = FGSLR_mu_set
for FGSLR_delta = FGSLR_delta_set
for idx_FGSLR_regul_B = idc_FGSLR_regul_B
FGSLR_regul_B = FGSLR_regul_B_set{idx_FGSLR_regul_B};

i = i + 1;


%% Loading restored HS images
name_method = 'FGSLR';
name_params_savetext = append('b', num2str(FGSLR_beta), '_m', num2str(FGSLR_mu), ...
        '_d', num2str(FGSLR_delta), '_', FGSLR_regul_B);


save_folder_name = append(...
    '../../result/' , ...
    'denoising_', image, '/', ...
    'g', num2str(deg.gaussian_sigma), '_ps', num2str(deg.sparse_rate), ...
        '_pt', num2str(deg.stripe_rate), '/', ...
    name_method, '/', ...
    name_params_savetext, '/' ...   
);

load(append(save_folder_name, 'image_result.mat'), ...
    'HSI_restored' ...
);

load(append(save_folder_name, 'metric_vals.mat'), ...
    'mpsnr', 'mssim', 'sam' ...
);


fprintf('%d:  \t %#.4g\t %#.4g\t %#.4g\n', i, mpsnr, mssim, sam);


if best_mpsnr < mpsnr
    best_mpsnr = mpsnr;

    best_beta = FGSLR_beta;
    best_mu = FGSLR_mu;
    best_delta = FGSLR_delta;
    best_regul_B = FGSLR_regul_B;
end



end
end
end
end


%% Loading best restored HS images
FGSLR_beta      = best_beta;
FGSLR_mu        = best_mu;
FGSLR_delta     = best_delta;
FGSLR_regul_B   = best_regul_B;


params_tmp = {FGSLR_beta, FGSLR_mu, FGSLR_delta, FGSLR_regul_B};
name_params = {'beta', 'mu', 'delta', 'regul_B'};
name_params_savetext = append('b', num2str(FGSLR_beta), '_m', num2str(FGSLR_mu), ...
        '_d', num2str(FGSLR_delta), '_', FGSLR_regul_B);


save_folder_name = append(...
    '../../result/' , ...
    'denoising_', image, '/', ...
    'g', num2str(deg.gaussian_sigma), '_ps', num2str(deg.sparse_rate), ...
        '_pt', num2str(deg.stripe_rate), '/', ...
    name_method, '/', ...
    name_params_savetext, '/' ...   
);

load(append(save_folder_name, 'image_result.mat'));
load(append(save_folder_name, 'metric_vals.mat'));
load(append(save_folder_name, 'other_result.mat'));


fprintf('~~~ BEST RESULTS ~~~\n');
fprintf('MPSNR: %#.4g, MSSIM: %#.4g, SAM: %#.4g\n', mpsnr, mssim, sam);
fprintf('beta: %g, mu: %g, delta: %g, norm: %s\n', ...
    FGSLR_beta, FGSLR_mu, FGSLR_delta, FGSLR_regul_B);


%% Saving best restored HS images
is_redir_best_params = 1;
if exist('is_redir_best_params', 'var') && is_redir_best_params
    save_best_folder_name = append(...
    '../../result/' , ...
    'denoising_', image, '/', ...
    'g', num2str(deg.gaussian_sigma), '_ps', num2str(deg.sparse_rate), ...
        '_pt', num2str(deg.stripe_rate), '/', ...
    name_method, '/', ...
    'best_params/' ...   
    );

    if exist(save_best_folder_name, 'dir')
        rmdir(save_best_folder_name, 's');
    end
end

save_folder_name_for_best = append(...
    '../../result/' , ...
    'denoising_', image, '/', ...
    'g', num2str(deg.gaussian_sigma), '_ps', num2str(deg.sparse_rate), ...
        '_pt', num2str(deg.stripe_rate), '/', ...
    name_method, '/', ...
    'best_params/', ...
    name_params_savetext, '/' ...   
);

mkdir(save_folder_name_for_best);

save(append(save_folder_name_for_best, 'image_result.mat'), ...
    'HSI_clean', 'HSI_noisy', 'hsi', 'deg', 'image', ...
    'HSI_restored', 'removed_noise', ...
    '-v7.3', '-nocompression' ...
);

save(append(save_folder_name_for_best, 'metric_vals.mat'), ...
    'mpsnr', 'mssim', 'ergas', 'sam', ...
    'psnr_per_band', 'ssim_per_band', ...
    '-v7.3', '-nocompression' ...
);

save(append(save_folder_name_for_best, 'other_result.mat'), ...
    'params', 'other_result', ...
    '-v7.3', '-nocompression' ...
);


end
end
