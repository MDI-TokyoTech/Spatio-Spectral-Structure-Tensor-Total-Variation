clear
close all;

addpath('../../func_metrics/')
addpath(genpath('../../sub_functions'))

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
total_exp = length(idc_noise_conditions);


for idx_image = idc_images
%% Generating observation
image = images{idx_image};
image_load_name = images_load_names{idx_image};

[HSI_noisy, hsi] = Load_real_HSI_for_Deep(image_load_name);

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
% FastHyMix_k_subspace = {4, 6, 8, 10, 12};
FastHyMix_k_subspace = {4, 8, 12};


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
fprintf('Parameter settings: %s\n', name_params_savetext)
fprintf('Cases: (%d/%d), Params:(%d/%d)\n', ...
    idx_exp, total_exp, idx_params_comb, num_params_comb);

[HSI_restored, removed_noise, other_result]...
    = func_methods{1}(params);


% Saving each result
save_folder_name = append(...
    '../../result/' , ...
    'denoising_real_HSI_', image, '/', ...
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

fprintf('******* finis *******\n');