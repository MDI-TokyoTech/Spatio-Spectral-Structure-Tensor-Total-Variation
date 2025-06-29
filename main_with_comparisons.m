%% Spatio-Spectral Structure Tensor Total Variation for Hyperspectral Image Denoising and Destriping
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Shingo Takemoto (takemoto.s.e908@m.isct.ac.jp)
% Last version: June 15, 2025
% Article: S. Takemoto, K. Naganuma, S. Ono, 
%   ``Spatio-Spectral Structure Tensor Total Variation for Hyperspectral Image Denoising and Destriping''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all
addpath(genpath('sub_functions'))
addpath('compared_methods')
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

HSI_clean = single(HSI_clean);
HSI_noisy = single(HSI_noisy);

edge_width = 3;


%% Setting common parameters
%%%%%%%%%%%%%%%%%%%%% User Settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%
rho = 0.95; % Parameter for the radii of the noise terms

blocksize = {[10,10]}; % Block size of S3TTV

stopcri_idx = 5; % Index of Stopping criterion
% maxiter = 20000; % Maximum number of iterations
maxiter = 5; % Maximum number of iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Stopping criterion
stopcri = 10 ^ -stopcri_idx; 

% Radius of v-centered l2-ball constraint serving data-fidelity
epsilon = rho * deg.Gaussian_sigma * sqrt(hsi.N * (1 - deg.sparse_rate));

% Radius of l1-ball constraint for sparse noise
alpha = rho * (0.5 * hsi.N * deg.sparse_rate);

% Radius of l1-ball constraint for stripe noise
beta = rho * hsi.N * (1 - deg.sparse_rate) ...
    * deg.stripe_rate * deg.stripe_intensity / 2; 


%% Setting each methods info
% S3TTV (ours)
methods_info(1) = struct( ...
    "name", "S3TTV", ...
    "func", @(HSI_noisy, params) S3TTV_GPU_fast(HSI_noisy, params), ...
    "param_names", {{"blocksize", "maxiter", "stopcri", "epsilon", "alpha", "beta"}}, ...
    "params", {{blocksize, maxiter, stopcri, epsilon, alpha, beta}}, ...
    "get_params_savetext", @(params) ...
        sprintf("bl%d_r%.2f_stop1e-%d", params.blocksize(1), rho, stopcri_idx), ...
    "enable", false ...
);

% SSTV
methods_info(end+1) = struct( ...
    "name", "SSTV", ...
    "func", @(HSI_noisy, params) SSTV_GPU(HSI_noisy, params), ...
    "param_names", {{"maxiter", "stopcri", "epsilon", "alpha", "beta"}}, ...
    "params", {{maxiter, stopcri, epsilon, alpha, beta}}, ...
    "get_params_savetext", @(params) ...
        sprintf("r%.2f_stop1e-%d", rho, stopcri_idx), ...
    "enable", false ...
);

% HSSTV_L1
HSSTV_omega = {0.05};

methods_info(end+1) = struct( ...
    "name", "HSSTV_L1", ...
    "func", @(HSI_noisy, params) HSSTV_GPU(HSI_noisy, params), ...
    "param_names", {{"L", "omega", "maxiter", "stopcri", "epsilon", "alpha", "beta"}}, ...
    "params", {{{"L1"}, HSSTV_omega, maxiter, stopcri, epsilon, alpha, beta}}, ...
    "get_params_savetext", @(params) ...
        sprintf("o%.2f_r%.2f_stop1e-%d", params.omega, rho, stopcri_idx), ...
    "enable", false ...
);

% HSSTV_L12
HSSTV_omega = {0.05};

methods_info(end+1) = struct( ...
    "name", "HSSTV_L12", ...
    "func", @(HSI_noisy, params) HSSTV_GPU(HSI_noisy, params), ...
    "param_names", {{"L", "omega", "maxiter", "stopcri", "epsilon", "alpha", "beta"}}, ...
    "params", {{{"L12"}, HSSTV_omega, maxiter, stopcri, epsilon, alpha, beta}}, ...
    "get_params_savetext", @(params) ...
        sprintf("o%.2f_r%.2f_stop1e-%d", params.omega, rho, stopcri_idx), ...
    "enable", false ...
);

% l0l1HTV
l0l1HTV_stepsize_reduction = 0.999;    
l0l1HTV_L10ball_th = {0.02, 0.03, 0.04};

methods_info(end+1) = struct( ...
    "name", "l0l1HTV", ...
    "func", @(HSI_noisy, params) l0l1HTV_GPU(HSI_noisy, params), ...
    "param_names", {{"L10ball_th", "stepsize_reduction", ...
        "maxiter", "stopcri", "epsilon", "alpha", "beta"}}, ...
    "params", {{l0l1HTV_L10ball_th, l0l1HTV_stepsize_reduction, ...
        maxiter, stopcri, epsilon, alpha, beta}}, ...
    "get_params_savetext", @(params) ...
        sprintf("sr%.5g_th%.2f_r%.2f_maxiter%d", ...
            params.stepsize_reduction, params.L10ball_th, rho, maxiter), ...
    "enable", false ...
);

% STV
methods_info(end+1) = struct( ...
    "name", "STV", ...
    "func", @(HSI_noisy, params) STV_GPU(HSI_noisy, params), ...
    "param_names", {{"blocksize", "maxiter", "stopcri", "epsilon", "alpha", "beta"}}, ...
    "params", {{blocksize, maxiter, stopcri, epsilon, alpha, beta}}, ...
    "get_params_savetext", @(params) ...
        sprintf("bl%d_r%.2f_stop1e-%d", params.blocksize(1), rho, stopcri_idx), ...
    "enable", false ...
);


% SSST
methods_info(end+1) = struct( ...
    "name", "SSST", ...
    "func", @(HSI_noisy, params) SSST_GPU(HSI_noisy, params), ...
    "param_names", {{"blocksize", "maxiter", "stopcri", "epsilon", "alpha", "beta"}}, ...
    "params", {{blocksize, maxiter, stopcri, epsilon, alpha, beta}}, ...
    "get_params_savetext", @(params) ...
        sprintf("bl%d_r%.2f_stop1e-%d", params.blocksize(1), rho, stopcri_idx), ...
    "enable", false ...
);


% LRTDTV
LRTDTV_tau = 1;
LRTDTV_lambda_param = {10, 15, 20, 25};
LRTDTV_lambda = cellfun(@(x) 100 * x / sqrt(100*100), LRTDTV_lambda_param);
LRTDTV_rank = {[hsi.n1*0.8, hsi.n2*0.8, 10]};

methods_info(end+1) = struct( ...
    "name", "LRTDTV", ...
    "func", @(HSI_noisy, params) func_LRTDTV(HSI_noisy, params), ...
    "param_names", {{"tau", "lambda", "rank"}}, ...
    "params", {{LRTDTV_tau, LRTDTV_lambda, LRTDTV_rank}}, ...
    "get_params_savetext", @(params) ...
        sprintf("l%.4g_r%d_stop1e-%d", params.lambda, params.rank(1), stopcri_idx), ...
    "enable", false ...
);

% FGSLR
FGSLR_beta_set = {0.1, 0.5};
FGSLR_mu_set = {5, 10};
FGSLR_delta_set = {0.5, 5};
FGSLR_regul_B_set = {'L2', 'L21'};
idc_FGSLR_regul_B = 1:numel(FGSLR_regul_B_set);

methods_info(end+1) = struct( ...
    "name", "FGSLR", ...
    "func", @(HSI_noisy, params) func_FGSLR(HSI_noisy, params), ...
    "param_names", {{"beta", "mu", "delta", "regul_B"}}, ...
    "params", {{FGSLR_beta_set, FGSLR_mu_set, FGSLR_delta_set, FGSLR_regul_B_set}}, ...
    "get_params_savetext", @(params) ...
        sprintf("b%.2g_m%d_d%0.2g_regulB%s", params.beta, params.mu, params.delta, params.regul_B), ...
    "enable", false ...
);

% TPTV
TPTV_Rank = {[7,7,5]};
TPTV_initial_rank = {2};
TPTV_maxIter = {50, 100};
TPTV_lambdas = {5e-4, 1e-4, 1e-3, 1e-2, 1.5e-2};

methods_info(end+1) = struct( ...
    "name", "TPTV", ...
    "func", @(HSI_noisy, params) func_TPTV(HSI_noisy, params), ...
    "param_names", {{"Rank", "initial_rank", "maxIter", "lambda"}}, ...
    "params", {{TPTV_Rank, TPTV_initial_rank, TPTV_maxIter, TPTV_lambdas}}, ...
    "get_params_savetext", @(params) ...
        sprintf("maxiter%d_l%.4g", params.maxIter, params.lambda), ...
    "enable", false ...
);

% FastHyMix
FastHyMix_k_subspace = {4,8,12};

methods_info(end+1) = struct( ...
    "name", "FastHyMix", ...
    "func", @(HSI_noisy, params) func_FastHyMix(HSI_noisy, params), ...
    "param_names", {{"k_subspace"}}, ...
    "params", {{FastHyMix_k_subspace}}, ...
    "get_params_savetext", @(params) ...
        sprintf("sub%g", params.k_subspace), ...
    "enable", false ...
);

methods_info = methods_info([methods_info.enable]); % removing false methods
num_methods = numel(methods_info);

i_method = 0;


%% Running methods
for idx_method = 1:num_methods
name_method = methods_info(idx_method).name;
func_method = methods_info(idx_method).func;
params_name = methods_info(idx_method).param_names;
params_cell = methods_info(idx_method).params;

[params_comb, num_params_comb] = ParamsList2Comb(params_cell);


for idx_params_comb = 1:num_params_comb

params = struct();
for idx_params = 1:numel(params_name)
    % Assigning parameters to the structure
    params.(params_name{idx_params}) = params_comb{idx_params_comb}{idx_params};
end

name_params_savetext = methods_info(idx_method).get_params_savetext(params);


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
