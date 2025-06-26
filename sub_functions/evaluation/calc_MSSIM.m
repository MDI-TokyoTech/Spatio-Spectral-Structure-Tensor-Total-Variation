function MSSIM = calc_MSSIM(restorated_HSI, org_HSI)
n3 = size(org_HSI, 3);

sum_ssim = 0;

for k = 1:n3
    sum_ssim = sum_ssim + ssim(org_HSI(:,:,k), restorated_HSI(:,:,k));
end

MSSIM = sum_ssim/n3;
