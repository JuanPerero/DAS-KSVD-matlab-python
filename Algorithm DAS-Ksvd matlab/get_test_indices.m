% select the indices to be used for testing.
function [out ] = get_test_indices( idx_train, n_studies )

    out = n_studies-length(idx_train);
    j = 1;
    for i = 1:n_studies
        if idx_train ~= i
            out(j) = i;
            j = j+1;
    end

end
