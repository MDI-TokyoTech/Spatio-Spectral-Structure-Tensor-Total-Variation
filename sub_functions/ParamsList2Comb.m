function [params_comb, num_params_comb] = ParamsList2Comb(params_list)

if all(cellfun(@(x) isscalar(x), params_list))
    % If all parameters have only one possible value
    params_comb = {params_list};
    num_params_comb = 1;

    for idx_params = 1:numel(params_comb{1})
        % If the content is a cell, extract the content
        if iscell(params_comb{1}{idx_params})
            params_comb{1}{idx_params} = params_comb{1}{idx_params}{1};
        end
    end
else

    % Compute the Cartesian product for all parameters
    [params_grid_tmp{1:numel(params_list)}] = ndgrid(params_list{:});
    params_grid_tmp = params_grid_tmp(1:numel(params_list));
    
    % Convert each grid to column vectors
    params_grid = cellfun(@(x) x(:), params_grid_tmp, 'UniformOutput', false);
    for j = 1:numel(params_grid)
        if ~iscell(params_grid{j})
            params_grid{j} = num2cell(params_grid{j});
        end
    end

    
    % Initialize params_cell and add index as the first element in each combination
    num_params_comb = numel(params_grid{1}); % Number of combinations
    params_comb = cell(num_params_comb, 1);
    
    % Adding the index and ensuring correct order of parameters
    for idx_params_comb = 1:num_params_comb
        params_comb{idx_params_comb} = [cellfun(@(x) x{idx_params_comb}, params_grid, 'UniformOutput', false)];

        for idx_params = 1:numel(params_list)
            % If the content is a cell, extract the content
            if iscell(params_comb{idx_params_comb}{idx_params})
                params_comb{idx_params_comb}{idx_params} = params_comb{idx_params_comb}{idx_params}{1};
            end
        end
    end

    % for idx_params = 1:numel(params_list)
    %     % If the content is a cell, extract the content
    %     if iscell(params_comb{idx_params_comb}{idx_params})
    %         params_comb{idx_params_comb}{idx_params} = params_comb{idx_params_comb}{idx_params}{1};
    %     end
    % end
end

end