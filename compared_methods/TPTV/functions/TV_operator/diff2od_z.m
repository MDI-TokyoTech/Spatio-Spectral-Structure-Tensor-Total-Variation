function tv_z = diff2od_z(x,sizeD)

tenX     = reshape(x, sizeD);
dfz1     = diff(tenX, 2, 3);
dfz      = zeros(sizeD);
dfz(:,:,1:end-2) = dfz1;
dfz(:,:,end-1)     = tenX(:,:,1) - 2*tenX(:,:,end) + tenX(:,:,end-1);
dfz(:,:,end)       = tenX(:,:,end) - 2*tenX(:,:,1) + tenX(:,:,2);
tv_z=dfz(:);

