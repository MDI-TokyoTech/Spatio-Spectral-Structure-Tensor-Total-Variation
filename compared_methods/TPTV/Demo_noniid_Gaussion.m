
% Demo on i.i.d. Gaussian Noise
clear,clc
datadir  = 'E:\Experiments\数据集\CAVE\matfile\';
fileFolder = fullfile(datadir);
dirOutput  = dir(fullfile(fileFolder,'*.mat'));
num_data = length(dirOutput);
dataname = cell(num_data,1);
noise_type = 'noniidGauss';

%% compare methods
methodname = {'None','BM4D','NMoG**','LRTV**','LRTDTV','NGmeet','WLRATV', 'ETV***','CTV***', 'WETV'};
num_method = length(methodname);

for k = 1 : num_data
    %% load data
    dataname{k}   = dirOutput(k).name;
    load(fullfile(datadir, dataname{k}))
    Ori_H = imresize(A,[200,200]);
    [M, N, B] = size(Ori_H);
    %% noise simulated
   Noi_H = Ori_H;
temph = reshape(Ori_H,M*N,B);
sigma_signal = sum(temph.^2)/(M*N);
SNR = 1 + rand(1,B)*15;
SNR1 = 10.^(SNR./10);
var_noi = sigma_signal./SNR1;
for b=1:B
    Noi_H(:,:,b) = Ori_H(:,:,b) + randn(M,N)*sqrt(var_noi(b));
end
NoisyHSI{k} = Noi_H;
    noise     = reshape(Noi_H - Ori_H, M*N,B);
    Var_noise  = var(noise);
    sigma_noi = sqrt(Var_noise);
    
     %% WETV
    disp('############## WETV #################')
    j=10;
    param.Rank   = [7,7,5];
    param.initial_rank = 2;
     param.maxIter = 50;
        param.lambda    = 4e-3*sqrt(M*N);  %0.1;   %mu2      = Alpha*mu1
        tic
        [ output_image,U_x,V_x,E] = WETV(Noi_H,Ori_H, param);
        Re_hsi_wETV   = reshape(output_image,M,N,B);
        Time  = toc;
        [~, mpsnr(k), ~ , mssim(k), egras(k),sam(k)] = evaluate_HSI(Ori_H,reshape(Re_hsi_wETV,M,N,B));
 
    
end


