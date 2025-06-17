function MPSNR = calc_MPSNR(HSI_restored, HSI_clean)
[n1, n2, n3] = size(HSI_clean);
difference_HSI = HSI_clean - HSI_restored;

psnr_per_band = 20*log10(sqrt(n1*n2) ./ reshape(sqrt(sum(difference_HSI.*difference_HSI, [1,2])), [n3, 1]));
MPSNR = mean(psnr_per_band);
