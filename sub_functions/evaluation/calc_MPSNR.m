function MPSNR = calc_MPSNR(restorated_HSI, org_HSI)
[n1, n2, n3] = size(org_HSI);
difference_HSI = org_HSI - restorated_HSI;

psnr_per_band = 20*log10(sqrt(n1*n2) ./ reshape(sqrt(sum(difference_HSI.*difference_HSI, [1,2])), [n3, 1]));
MPSNR = mean(psnr_per_band);
