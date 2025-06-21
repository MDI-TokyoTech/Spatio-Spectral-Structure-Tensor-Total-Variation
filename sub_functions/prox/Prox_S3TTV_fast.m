function[z_output] = Prox_S3TTV_fast(z, gamma, blocksize)
% z = gather(z);
[n1, n2, n3, d] = size(z);

z_output = zeros(size(z), 'single', 'gpuArray');

for i = 1:n1
    rows = mod((i:i+blocksize(1)-1)-1, n1) + 1;

    for j = 1:n2
        cols = mod((j:j+blocksize(2)-1)-1, n2) + 1;

        block_tensor = z(rows, cols, : ,:);
        M = reshape(block_tensor, [prod(blocksize), n3*d]);
        
        [U, S, V] = svd(M,'econ');
        Sthre = diag(max(0, diag(S) - gamma));
        z_output(rows, cols, : ,:) ...
            = z_output(rows, cols, : ,:) + reshape(U*Sthre*V', [blocksize, n3, d]);
    end
end


% z = gpuArray(z);