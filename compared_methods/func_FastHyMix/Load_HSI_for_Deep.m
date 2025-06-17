function  [HSI_clean, hsi] = Load_HSI_for_Deep(set_image)
rng('default')
%% Loading HSI
switch set_image
    case 'JasperRidge'
        load('../../../HSIData/JasperRidge/jasperRidge2_R198.mat');
        U_tmp = reshape(Y, [198, 100, 100]);
        HSI_clean = normalize01(permute(U_tmp, [2,3,1]));

    case 'PaviaU120'
        load('../../../HSIData/PaviaU/PaviaU.mat')
        hsi.start_pos = [170, 210, 5];
        hsi_size = [120, 120, size(paviaU, 3)-hsi.start_pos(3)+1];
        hsi.end_pos = hsi.start_pos + hsi_size - 1;
        HSI_clean = normalize01(paviaU(hsi.start_pos(1):hsi.end_pos(1), hsi.start_pos(2):hsi.end_pos(2), hsi.start_pos(3):hsi.end_pos(3)));
        
    case 'Beltsville'
        load('../../../HSIData/Beltsville.mat');
        image = "Beltsville";
        org_image = u_org;
        start_pos = [145, 157, 1];
        HSI_size = [100, 100, size(org_image, 3)];
        end_pos = start_pos + HSI_size - 1;
        org_HSI_tmp = org_image(start_pos(1):end_pos(1), start_pos(2):end_pos(2), start_pos(3):end_pos(3));
        HSI_clean = normalize01(org_HSI_tmp);

end


hsi.sizeof = size(HSI_clean);
[hsi.n1,hsi.n2,hsi.n3] = size(HSI_clean);
hsi.N = prod(hsi.sizeof);
