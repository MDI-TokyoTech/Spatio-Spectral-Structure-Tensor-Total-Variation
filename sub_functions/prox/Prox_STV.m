function[z] = Prox_STV(z, gamma, blocksize)
z = gather(z);
[v, h, c, d, s1, s2] = size(z);

for i = 1:v/blocksize(1)
    for j = 1:h/blocksize(2)
        for k = 1:s1
            for l = 1:s2
                block_tensor = z(1+blocksize(1)*(i-1):blocksize(1)*i, 1+blocksize(2)*(j-1):blocksize(2)*j, : ,:, k, l);
                M = reshape(block_tensor, [prod(blocksize)*c, d]);
                
                [U, S, V] = svd(M,'econ');
                Sthre = diag(max(0, diag(S) - gamma));
                z(1+blocksize(1)*(i-1):blocksize(1)*i,1+blocksize(2)*(j-1):blocksize(2)*j,:,:,k,l)...
                    = reshape(U*Sthre*V', [blocksize, c, d]);
            end
        end
    end
end

z = gpuArray(z);