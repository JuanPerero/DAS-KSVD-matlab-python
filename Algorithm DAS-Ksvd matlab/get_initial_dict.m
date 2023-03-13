function [ init_dict ] = get_initial_dict( data, red_factor, type )
    
    % data:
    % red_factor:
    % type: ->1 (random) ->0 (from data)

    [sampledim,num_samples] = size(data);
   
    if type == 1 % random
        initialdict = rand(sampledim,red_factor*sampledim)*20; % random initialization
        init_dict = normc(initialdict); % column normalization
    elseif type == 0 % from data
        examples_ini = randperm(num_samples,red_factor*sampledim);
        initialdict = data(:,examples_ini);
        init_dict = normc(initialdict);
    end
   
end