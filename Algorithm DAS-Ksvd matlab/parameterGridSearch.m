function [ parameters, n_parameters ] = parameterGridSearch(points, graph)

    parameters = [];
    alpha = 0:1/(points-1):1;
    beta = alpha;
    if graph
        figure()
        hold on
    end
    for p1 = 1:points
        alpha_new = alpha(p1);
        for p2 = 1:points
            beta_new = beta(p2);
            if alpha_new+beta_new<=1
                gamma_new = abs(1-alpha_new-beta_new);
                parameters = [parameters;[alpha_new,beta_new,gamma_new]];
                if graph
                    plot(alpha_new,beta_new,'r*')
                end
            end
        end
    end
    n_parameters = length(parameters);

end
