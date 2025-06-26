clear;
close all;

addpath(genpath("sub_functions"))
addpath("func_metrics")


%% Switching operator
% is_show_HSI = 1;
diff_magnification = 7;

% is_plot_psnr_and_ssim_per_band = 1;


%% Selecting conditions
noise_conditions = { ...
    {0.1,   0.05,  0,     0,    0},     ... % g0.1 ps0.05 pt0 pd0
    {0.1,   0.1,   0,     0,    0},     ... % g0.1 ps0.05 pt0 pd0
    {0.1,   0,     0,     0,    0},     ... % g0.1 ps0 pt0
    {0,     0,     0.05,  0.5,  0},     ... % g0 ps0 pt0.05
    {0.05,  0.05,  0,     0,    0},     ... % g0.05 ps0.05 pt0
    {0.1,   0.05,  0,     0,    0},     ... % g0.1 ps0.05 pt0
    {0.05,  0,     0.05,  0.5,  0},     ... % g0.05 ps0 pt0.05
    {0.1,   0,     0.05,  0.5,  0},     ... % g0.1 ps0 pt0.05
    {0.05,  0.05,  0.05,  0.3,  0},     ... % g0.05 ps0.05 pt0.05
    {0.1,   0.05,  0.05,  0.3,  0},     ... % g0.1 ps0.05 pt0.05
    {0.05,  0.05,  0.05,  0.5,  0},     ... % g0.05 ps0.05 pt0.05
    {0.1,   0.05,  0.05,  0.5,  0},     ... % g0.1 ps0.05 pt0.05
    {0.05,  0.05,  0.05,  0.3,  0.001}, ... % g0.05 ps0.05 pt0.05 pd0.001
    {0.1,   0.05,  0.05,  0.3,  0.001}, ... % g0.1 ps0.05 pt0.05 pd0.001
};

% idc_noise_conditions = 1:size(noise_conditions, 2);
idc_noise_conditions = [5:6, 11:14];
% idc_noise_conditions = 14;

images = {...
    "JasperRidge", ...
    "PaviaU", ...
    "Beltsville", ...
};

idc_images = 1:numel(images);
% idc_images = 1;


%% Setting common parameters
% switch_rho = "0.93"; % "0.93", "0.95", "0.98", "all"
switch_rho = "0.95";
% switch_rho = "0.98";
% switch_rho = "all";

stopcri_idx = 5;
stopcri = 10 ^ -stopcri_idx;

maxiter = 20000;
% maxiter = 5;


%% Setting each methods info
% SSTV
methods_info(1) = struct( ...
    "name", "SSTV", ...
    "is_best_param", "each_rho", ...
    "line_style", "--", ...
    "enable", true ...
);

% HSSTV_L1
methods_info(end+1) = struct( ...
    "name", "HSSTV_L1", ...
    "is_best_param", "each_rho", ...
    "line_style", "--", ...
    "enable", true ...
);

% HSSTV_L12
methods_info(end+1) = struct( ...
    "name", "HSSTV_L12", ...
    "is_best_param", "each_rho", ...
    "line_style", "--", ...
    "enable", true ...
);

% HSSTV4
methods_info(end+1) = struct( ...
    "name", "HSSTV4", ...
    "is_best_param", "each_rho", ...
    "line_style", "--", ...
    "enable", true ...
);

% LRTDTV
methods_info(end+1) = struct( ...
    "name", "LRTDTV", ...
    "is_best_param", "all", ...
    "line_style", "--", ...
    "enable", true ...
);

% TPTV
methods_info(end+1) = struct( ...
    "name", "TPTV", ...
    "is_best_param", "all", ...
    "line_style", "--", ...
    "enable", true ...
);

% RISSTV
methods_info(end+1) = struct( ...
    "name", "RISSTV", ...
    "is_best_param", "each_rho", ...
    "line_style", "-", ...
    "enable", true ...
);

% TdSSTV_L1
methods_info(end+1) = struct( ...
    "name", "TdSSTV_L1", ...
    "is_best_param", "each_rho", ...
    "line_style", "-", ...
    "enable", true ...
);

% TdSSTV_L12
methods_info(end+1) = struct( ...
    "name", "TdSSTV_L12", ...
    "is_best_param", "each_rho", ...
    "line_style", "-", ...
    "enable", true ...
);


methods_info = methods_info([methods_info.enable]); % removing false methods
num_methods = numel(methods_info);


%% Loading results
for idx_noise_condition = idc_noise_conditions
for idx_image = idc_images
%% Selecting noise condition
deg.gaussian_sigma      = noise_conditions{idx_noise_condition}{1};
deg.sparse_rate         = noise_conditions{idx_noise_condition}{2};
deg.stripe_rate         = noise_conditions{idx_noise_condition}{3};
deg.stripe_intensity    = noise_conditions{idx_noise_condition}{4};
deg.deadline_rate       = noise_conditions{idx_noise_condition}{5};
image = images{idx_image};


[HSI_clean, hsi] = Load_HSI(image);
noise_seed = "default";
[HSI_noisy, deg] = Generate_obsv_for_denoising(HSI_clean, deg, noise_seed);

HSI_clean = single(HSI_clean);
HSI_noisy = single(HSI_noisy);


%% Getting save_text for methods Requiging best params
load("save_result_dir.mat", "save_result_dir");

for idx_method = 1:num_methods
    switch methods_info(idx_method).is_best_param
        case "all"
            save_best_folder_name = append(...
                save_result_dir, ...
                "denoising_", image, "/", ...
                "g", num2str(deg.gaussian_sigma), "_ps", num2str(deg.sparse_rate), ...
                        "_pt", num2str(deg.stripe_rate), "_pd", num2str(deg.deadline_rate), "/", ...
                methods_info(idx_method).name, "/" ...
            );
    
            load(append(save_best_folder_name, "best_params.mat"), "best_params_savetext");
    
            methods_info(idx_method).get_params_savetext = best_params_savetext;
    
    
        case "each_rho"
            save_best_folder_name = append(...
                save_result_dir, ...
                "denoising_", image, "/", ...
                "g", num2str(deg.gaussian_sigma), "_ps", num2str(deg.sparse_rate), ...
                        "_pt", num2str(deg.stripe_rate), "_pd", num2str(deg.deadline_rate), "/", ...
                methods_info(idx_method).name, "/" ...
            );
    
            if strcmp(switch_rho, "all")
                load(append(save_best_folder_name, "best_params.mat"), "best_params_savetext");
            else
                load(append(save_best_folder_name, "best_params_r", switch_rho, ".mat"), "best_params_savetext");
            end
    
            methods_info(idx_method).get_params_savetext = best_params_savetext;
    
        otherwise
            error("Invalid value for is_best_param: %s", methods_info(idx_method).is_best_param);
    end
end




%% Loading each result
% Initialization
list_psnr_per_band = [];
list_ssim_per_band = [];

name_length_max = max(strlength([methods_info.name]));
name_params_length_max = max(strlength([methods_info.get_params_savetext]));


fprintf("~~~ SETTINGS ~~~\n");
fprintf("Image: %s Size: (%d, %d, %d)\n", image, hsi.n1, hsi.n2, hsi.n3);
fprintf("Gaussian sigma: %g\n", deg.gaussian_sigma);
fprintf("Sparse rate: %g\n", deg.sparse_rate);
fprintf("Stripe rate: %g\n", deg.stripe_rate);
fprintf("Stripe intensity: %g\n", deg.stripe_intensity);
fprintf("Deadline rate: %g\n", deg.deadline_rate);

fprintf("~~~ RESULTS ~~~\n");
fprintf("%s  \t MPSNR\t MSSIM\t SAM\n", blanks(name_length_max + name_params_length_max + 2));


% Calculation noisy results
cat_HSI = cat(2, HSI_clean, HSI_noisy);
cat_diff = cat(2, zeros(hsi.sizeof), abs(HSI_clean - HSI_noisy));

val_mpsnr_noisy  = calc_MPSNR(HSI_noisy, HSI_clean);
val_mssim_noisy  = calc_MSSIM(HSI_noisy, HSI_clean);
val_sam_noisy    = calc_SAM(HSI_noisy, HSI_clean);


fprintf("%s: \t %#.4g\t %#.4g\t %#.4g\n", ...
    append("noisy", blanks(name_length_max + name_params_length_max - 3)), ...
    val_mpsnr_noisy, val_mssim_noisy, val_sam_noisy);


% Loading restored results
for idx_method = 1:num_methods
name_method = methods_info(idx_method).name;
params_save_text = methods_info(idx_method).get_params_savetext;

save_folder_name = append(...
    save_result_dir , ...
    "denoising_", image, "/", ...
    "g", num2str(deg.gaussian_sigma), "_ps", num2str(deg.sparse_rate), ...
        "_pt", num2str(deg.stripe_rate), "_pd", num2str(deg.deadline_rate), "/", ...
    name_method, "/", ...
    params_save_text, "/" ...   
);

load(append(save_folder_name, "image_result.mat"), ...
    "HSI_restored" ...
);

load(append(save_folder_name, "metric_vals.mat"), ...
    "val_mpsnr", "val_mssim", "val_sam", "vals_psnr_per_band", "vals_ssim_per_band" ...
);


cat_HSI = cat(2, cat_HSI, HSI_restored);
cat_diff = cat(2, cat_diff, abs(HSI_clean - HSI_restored));

list_psnr_per_band = cat(1, list_psnr_per_band, vals_psnr_per_band);
list_ssim_per_band = cat(1, list_ssim_per_band, vals_ssim_per_band);

fprintf("%s(%s): \t %#.4g\t %#.4g\t %#.4g\n", ...
    append(name_method, blanks(name_length_max - strlength(name_method))), ...
    append(params_save_text, blanks(name_params_length_max - strlength(params_save_text))), ...
    val_mpsnr, val_mssim, val_sam);

end


%% Showing HSI
if exist("is_show_HSI", "var") && is_show_HSI == 1
cat_diff = cat_diff * diff_magnification;
show_HSI = cat(1, cat_HSI, cat_diff);
implay(show_HSI)
end


%% Plotting psnr and ssim per band
if exist("is_plot_psnr_and_ssim_per_band", "var") && is_plot_psnr_and_ssim_per_band == 1
label_style = {"FontSize", 15};


figure(1)
hold on
for idx_method = 1:num_methods
    name_method = methods_info(idx_method).name;
    lineStyle_method = methods_info(idx_method).line_style;
    plot_style = {"LineWidth", 2.0, "LineStyle", lineStyle_method, "DisplayName", name_method};
    plot(list_psnr_per_band(idx_method,:), plot_style{:})
end
hold off
title("psnr per band")
xlabel("band", label_style{:})
ylabel("psnr", label_style{:})

figure(2)
hold on
for idx_method = 1:num_methods
    name_method = names_methods{idx_method};
    lineStyle_method = methods_info(idx_method).line_style;
    plot_style = {"LineWidth", 2.0, "LineStyle", lineStyle_method, "DisplayName", name_method};
    plot(list_ssim_per_band(idx_method,:), plot_style{:})
end
hold off
title("ssim per band")
xlabel("band", label_style{:})
ylabel("psnr", label_style{:})

end

end
end
