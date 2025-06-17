function MSSIM = calc_MSSIM(HSI_restored, HSI_clean)
n3 = size(HSI_clean, 3);

sum_ssim = 0;

for k = 1:n3
    sum_ssim = sum_ssim + ssim(HSI_clean(:,:,k), HSI_restored(:,:,k));
end

MSSIM = sum_ssim/n3;
