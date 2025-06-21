% 小野先生の記事から参照
% l1ノルムの近接写像
function result = Prox_l1norm(A, gamma)
    %{
    size_of_1st_way = size(A, 1);
    size_of_2nd_way = size(A, 2);
    size_of_3rd_way = size(A, 3);
    tmp = A;
    for k = 1:size_of_3rd_way
        for j = 1:size_of_2nd_way
            for i = 1:size_of_1st_way
                tmp(i, j, k) = sign(A(i, j, k))*max(abs(A(i, j, k))-gamma, 0);
            end
        end
    end
    result = tmp;
    %}
    result = sign(A).*max(abs(A) - gamma, 0);
end
