clear
close all;

addpath('func_metrics')

fprintf('******* initium *******\n');
rng('default')

%% Selecting conditions
images = {...
    'Swannee', ...
    'IndianPines', ...
};

images_load_names = {...
    'Swannee_401_220', ...
    'IndianPines120', ...
};

idc_images = 1:numel(images);
% idc_images = 1;

idx_exp = 0;
total_exp = length(idc_images);


for idx_image = idc_images
%% Generating observation
image = images{idx_image};
image_load_name = images_load_names{idx_image};

[HSI_noisy, hsi] = Load_real_HSI_for_refcode(image_load_name);

% HSI_clean = single(HSI_clean);
% HSI_noisy = single(HSI_noisy);

idx_exp = idx_exp + 1;


%% Selecting methods
name_method = 'FGSLR';

func_methods = {...
    @(params) func_FGSLR_real_HSI(HSI_noisy, params), ...
};


%% Setting parameters
% FGSLR
FGSLR_beta_set = [0.1, 0.5];
FGSLR_mu_set = [5, 10];
FGSLR_delta_set = [0.5, 5];
FGSLR_regul_B_set = {'L2', 'L21'};
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


% Saving each result
save_folder_name = append(...
    '../../result/' , ...
    'denoising_', image, '/', ...
    name_method, '/', ...
    name_params_savetext, '/' ...   
);

mkdir(save_folder_name);

save(append(save_folder_name, 'image_result.mat'), ...
    'HSI_noisy', 'hsi', 'image', ...
    'HSI_restored', 'removed_noise', ...
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
end

end

fprintf('******* finis *******\n');