function [psnr_per_band, ssim_per_band] = calc_PSNR_SSIM_per_band(HSI_restored, HSI_clean)
n3 = size(HSI_clean, 3);
psnr_per_band = zeros(1,n3);
ssim_per_band = zeros(1,n3);
for l = 1:n3
psnr_per_band(l) = psnr(HSI_restored(:,:,l), HSI_clean(:,:,l));
ssim_per_band(l) = ssim(HSI_restored(:,:,l), HSI_clean(:,:,l));
end