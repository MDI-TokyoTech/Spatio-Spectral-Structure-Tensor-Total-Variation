function [HSI_restored, removed_noise, other_result] = func_FastHyMix(HSI_noisy, params)
addpath('scripts');
k_subspace = params.k_subspace;

img_noisy = HSI_noisy;

tic;
[img_FastHyMix, M, noise_std_Gaussion] = FastHyMix(img_noisy,  k_subspace);

runtime = toc;
fprintf("Runtime: %g sec\n\n", runtime);

%% Organizing results for output
HSI_restored = img_FastHyMix;

removed_noise.all_noise = HSI_noisy - HSI_restored;

other_result.M = M;
other_result.noise_std_Gaussion = noise_std_Gaussion;
end

