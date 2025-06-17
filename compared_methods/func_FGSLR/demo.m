clc,clear,close all
addpath quality_assess
load('Pavia_80.mat'); 
Img = OriData3;
[m,n,b] = size(Img);
sigma = 0.15;
Img = Img./repmat(max(max(Img,[],1),[],2),m,n);
Noisy_Img = Img + sigma*randn(m,n,b);
Comparison_FGSLR_L2 = 1;
Comparison_FGSLR_GS = 0;
%% FGSLR-2/3
if Comparison_FGSLR_L2
      addpath FGSLR_PAM
      [M,N,p] = size(Noisy_Img);
      % For the setting of parameters, please refer to the discussion section of the article.
      opt.alpha = 1;
      opt.beta = 0.1;          % TV regularization:set as 0.1 or 0.5     
      opt.lambda = 0.01;       % sparse noise
      opt.mu = 5;              % penalty parameter:set as 5 or 10
      opt.rho = 0.1;           % proximal parameter
      opt.delta = 5;           % noise parameter:set as 0.5 or 5
      opt.r = 20;
      opt.regul_B = 'L2';
      [X, A, B] = FGSLR_TV_PAM(Noisy_Img, Img, opt);
      FGSLR_L2 = reshape(X',M,N,p);
      [psnr, ssim, fsim, ergas, msam] = MSIQA(Img*255, FGSLR_L2*255);
end
%% FGSLR-1/2
if Comparison_FGSLR_GS
      addpath FGSLR_PAM
      [M,N,p] = size(Noisy_Img);
      % For the setting of parameters, please refer to the discussion section of the article.
      opt.alpha = 1;
      opt.beta = 0.1;          % TV regularization:set as 0.1 or 0.5  
      opt.lambda = 0.01;       % sparse noise
      opt.mu = 5;              % penalty parameter:set as 5 or 10
      opt.rho = 0.1;           % proximal parameter
      opt.delta = 0.5;         % noise parameter:set as 0.5 or 5
      opt.r = 20;
      opt.regul_B = 'L21';
      [X, A, B] = FGSLR_TV_PAM(Noisy_Img, Img, opt);
      FGSLR_GS = reshape(X',M,N,p);
      [psnr, ssim, fsim, ergas, msam] = MSIQA(Img*255,FGSLR_GS*255);
end