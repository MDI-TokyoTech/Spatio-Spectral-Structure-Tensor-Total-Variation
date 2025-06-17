clear;
close all;

addpath(genpath('sub_functions'))


%% Selecting conditions
noise_conditions = { ...
    % {0.1, 0, 0, 0}, ... % g0.1 ps0 pt0
    % {0, 0, 0.05, 0.5}, ... % g0 ps0 pt0.05
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
};

idc_images = 1:numel(images);
% idc_images = 1;


num_noise_conditions = length(idc_noise_conditions);
num_images = length(idc_images);


%% Setting parameters
% FGSLR
FGSLR_beta_set = [0.1, 0.5];
FGSLR_mu_set = [5, 10];
FGSLR_delta_set = [0.5, 5];
FGSLR_regul_B_set = {'L2', 'L21'};
idc_FGSLR_regul_B = 1:numel(FGSLR_regul_B_set);

num_params = length(FGSLR_beta_set) * length(FGSLR_mu_set) * ...
                length(FGSLR_delta_set) * length(FGSLR_regul_B_set);

mpsnr_list = zeros(num_params+1, num_noise_conditions, num_images);
mssim_list = zeros(num_params+1, num_noise_conditions, num_images);
method_index_list = strings(num_params+1, 1);


%% Loading noise conditions
for idx_noise_condition = idc_noise_conditions
for idx_image = idc_images
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


i = 0;


%% Loading FGSLR result
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


load(append(save_folder_name, 'metric_vals.mat'), ...
    'mpsnr', 'mssim' ...
);


mpsnr_list(i, idx_noise_condition, idx_image) = mpsnr;
mssim_list(i, idx_noise_condition, idx_image) = mssim;
method_index_list(i) = name_params_savetext;


end
end
end
end

i = i + 1;


%% Loading S3TTV result
name_method = 'S3TTV';
rho = 0.95;

blocksize = [10,10]; % block size of STV-type methods
shiftstep = [1,1]; % [1,1] means full overlap

stopcri_index = 5;



% S3TTV
save_S3TTV_Calc_FV_folder_name = append(...
    '../../result/' , ...
    'denoising_', image, '/', ...
    'g', num2str(deg.gaussian_sigma), '_ps', num2str(deg.sparse_rate), ...
        '_pt', num2str(deg.stripe_rate), '/', ...
    'S3TTV/Calc_FV' ...
);

% Check if Calc_FV folder exists
if isfolder(save_S3TTV_Calc_FV_folder_name)
    name_param_save_text_S3TTV = ...
        append('Calc_FV/', ...
            'bl', num2str(blocksize(1)), '_st', num2str(shiftstep(1)), ...
            '_r', num2str(rho), '_stop1e-', num2str(stopcri_index) ...
        );
else
    name_param_save_text_S3TTV = ...
        append(...
            'bl', num2str(blocksize(1)), '_st', num2str(shiftstep(1)), ...
            '_r', num2str(rho), '_stop1e-', num2str(stopcri_index) ...
        );
end


save_folder_name = append(...
    '../../result/' , ...
    'denoising_', image, '/', ...
    'g', num2str(deg.gaussian_sigma), '_ps', num2str(deg.sparse_rate), ...
        '_pt', num2str(deg.stripe_rate), '/', ...
    name_method, '/', ...
    name_param_save_text_S3TTV, '/' ...   
);

load(append(save_folder_name, 'metric_vals.mat'), ...
    'mpsnr', 'mssim' ...
);

mpsnr_list(i, idx_noise_condition, idx_image) = mpsnr;
mssim_list(i, idx_noise_condition, idx_image) = mssim;
method_index_list(i) = name_method;


end
end

for i = 1:num_params+1
    if strlength(method_index_list(i)) ~= max(strlength(method_index_list), [], 'all')
        num_blanks = max(strlength(method_index_list), [], 'all') - strlength(method_index_list(i));
        method_index_list(i) = append(method_index_list(i), blanks(num_blanks));
    end
end

%% Plotting result
for idx_image = idc_images
fprintf('~~~ RESULTS ~~~\n');
fprintf('Image: %s\n\n', images{idx_image});

fprintf('MPSNR\n')
fprintf(append('Method', blanks(max(strlength(method_index_list), [], 'all') - 6), ...
    '\t'))

for idx_noise_condition = idc_noise_conditions
    fprintf('Case %d\t', idx_noise_condition);
end
fprintf('\n');

for i = 1:num_params+1
    fprintf('%s \t', method_index_list(i))
    for idx_noise_condition = idc_noise_conditions
        fprintf('%#.4g \t', mpsnr_list(i, idx_noise_condition, idx_image))
    end
    fprintf('\n')
end

fprintf('\n')

fprintf('MSSIM\n')
fprintf(append('Method', blanks(max(strlength(method_index_list), [], 'all') - 6), ...
    '\t'))

for idx_noise_condition = idc_noise_conditions
    fprintf('Case %d\t', idx_noise_condition);
end
fprintf('\n');

for i = 1:num_params+1
    fprintf('%s \t', method_index_list(i))
    for idx_noise_condition = idc_noise_conditions
        fprintf('%#.4g \t', mssim_list(i, idx_noise_condition, idx_image))
    end
    fprintf('\n')
end

fprintf('\n')

end