X = TPTV_restored_HSI;
Y = X + E;

noise = abs(HSI_clean - HSI_noisy);
% noise = HSI_clean - HSI_noisy;
% noise_01 = (noise - min(noise,[],"all")) / (max(noise,[],"all") - min(noise,[],"all"));

% E_01 = (E - min(E,[],"all")) / (max(E,[],"all") - min(E,[],"all"));

res_Y = abs(HSI_noisy - Y);
res_X = abs(HSI_clean - X);
res_E = abs(noise - abs(E));
% res_E = abs(noise_01 - E_01);


cat_obsv = cat(2, HSI_clean, HSI_noisy, noise);
cat_result = cat(2, X, Y, abs(E));
cat_res = cat(2, res_X, res_Y, res_E);

implay(cat(1, cat_obsv, cat_result, cat_res*7));