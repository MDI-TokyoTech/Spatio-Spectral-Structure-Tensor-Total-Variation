function  [HSI_noisy, hsi] = Load_real_HSI_for_Deep(set_image)
%% Loading HSI
switch set_image
    case 'IndianPines120'
        load('../../../HSIData/IndianPine/Indian_pines.mat')
        HSI_noisy = normalize01(indian_pines(1:end-25, 26:end, :));

    case 'Swannee_401_220'
        load('../../../HSIData/Suwannee_original.mat')
        start_pos = [401, 220];
        U_tmp = u_org(start_pos(1):start_pos(1)+99, start_pos(2):start_pos(2)+99, :);
        HSI_noisy = normalize01_by_band(U_tmp);
end

hsi.sizeof = size(HSI_noisy);
[hsi.n1,hsi.n2,hsi.n3] = size(HSI_noisy);
hsi.N = prod(hsi.sizeof);