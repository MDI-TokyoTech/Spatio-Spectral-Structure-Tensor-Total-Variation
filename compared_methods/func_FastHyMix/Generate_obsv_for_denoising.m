function[HSI_noisy, deg] = Generate_obsv_for_denoising(HSI_clean, deg)
[n1, n2, n3] = size(HSI_clean);

gaussian_sigma      = deg.gaussian_sigma;
sparse_rate         = deg.sparse_rate;
stripe_rate         = deg.stripe_rate;
% stripe_sigma        = deg.stripe_sigma;
stripe_intensity    = deg.stripe_intensity;


%% Generating stripe noise
if stripe_intensity > 0
    % warm_stripe only
    % stripe_noise = stripe_sigma*randn(1, n2, n3).*ones(n1, n2, n3);

    % sparse_stripe only
    sparse_stripe = 2*(imnoise(0.5*ones(1, n2, n3), "salt & pepper", stripe_rate) - 0.5).* ...
        rand(1, n2, n3).*ones(n1, n2, n3);
    stripe_noise = stripe_intensity.*sparse_stripe./max(abs(sparse_stripe), [], "all");


    % % warm_stripe + sparse_stripe
    % sparse_stripe = 2*(imnoise(0.5*ones(1, n2, n3), "salt & pepper", stripe_rate) - 0.5).* ...
    %     rand(1, n2, n3).*ones(n1, n2, n3);
    % warm_stripe = stripe_sigma*randn(1, n2, n3).*ones(n1, n2, n3);
    % stripe_noise = warm_stripe + sparse_stripe;
    % stripe_noise = stripe_intensity.*stripe_noise./max(abs(stripe_noise), [], "all");

    HSI_noisy = HSI_clean + stripe_noise;

    deg.stripe_noise        = stripe_noise;

elseif stripe_intensity == 0
    HSI_noisy = HSI_clean;
else
    disp('invalid value for is_stripe');
end


%% Generating Gaussian noise
if gaussian_sigma > 0
    gaussian_noise = gaussian_sigma*randn(n1, n2, n3);
    HSI_noisy = HSI_noisy + gaussian_noise;

    deg.gaussian_noise = gaussian_noise;

elseif gaussian_sigma == 0
else
    disp('invalid value for is_gaussian');
end


%% Generating sparse noise
if sparse_rate > 0
    HSI_tmp = HSI_noisy;

    Sp = 0.5*ones(n1, n2, n3);
    Sp = imnoise(Sp,'salt & pepper',sparse_rate);
    
    HSI_noisy(Sp==0) = 0;
    HSI_noisy(Sp==1) = 1;

    deg.sparse_noise = HSI_noisy - HSI_tmp;
    
elseif sparse_rate == 0
else
    disp('invalid value for is_sparse');
end